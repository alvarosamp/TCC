r"""
Script 3 — Baixar 1.358 event waveforms do SCEDC, filtrando inline.
ESTRUTURA HÍBRIDA:
  - RAW (events, xmls, catalog) -> G:\Meu Drive\TCC\data2\   (backup automático)
  - PROCESSED (.npz) e LOGS     -> C:\TCC_data\               (rápido, local)

Para cada evento:
  1. baixa .ms bruto do S3 em memória
  2. abre com obspy
  3. mantém só traces BHZ de estações que estão em stationxml/
  4. salva .ms enxuto em raw/events/{ano}/{evid}.ms
  5. loga estatísticas

- 8 downloads em paralelo (threads)
- Retomada: pula arquivos já baixados
- Logging em arquivo + console
"""

import csv
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
import obspy
from botocore import UNSIGNED
from botocore.config import Config

# ---------- Config: estrutura híbrida ----------
DRIVE_BASE = Path(r"G:\Meu Drive\TCC\data2")
LOCAL_BASE = Path(r"C:\TCC_data")

CSV_EVENTOS = DRIVE_BASE / "catalog" / "eventos_alvo.csv"
DIR_XML = DRIVE_BASE / "raw" / "stationxml"
DIR_EVENTS = DRIVE_BASE / "raw" / "events"
DIR_LOGS = LOCAL_BASE / "logs"   # logs ficam locais (escrita frequente)

BUCKET = "scedc-pds"
N_THREADS = 6                    # 6 em vez de 8 — Drive é mais lento, evita gargalo
CANAL_ALVO = "BHZ"

# ---------- Logging ----------
DIR_LOGS.mkdir(parents=True, exist_ok=True)
log_file = DIR_LOGS / f"download_events_{int(time.time())}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dl_events")

# ---------- Stats globais ----------
stats = {
    "ok": 0,
    "ja_existia": 0,
    "sem_bhz_valido": 0,
    "erro_download": 0,
    "erro_parse": 0,
    "bytes_baixados": 0,
    "bytes_salvos": 0,
}
stats_lock = Lock()


def carregar_estacoes_validas() -> set[str]:
    """Lê os XMLs e retorna set de 'NET.STA' válidos."""
    estacoes = set()
    for xml in DIR_XML.glob("*.xml"):
        nome = xml.stem
        if "." in nome:
            estacoes.add(nome)
    return estacoes


def carregar_eventos() -> list[dict]:
    eventos = []
    with open(CSV_EVENTOS, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eventos.append(row)
    return eventos


def fazer_cliente_s3():
    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=N_THREADS * 2,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def baixar_e_filtrar_evento(s3, evento: dict, estacoes_validas: set[str]) -> str:
    evid = evento["evid"]
    ano = int(evento["ano"])
    doy = int(evento["doy"])

    dest_dir = DIR_EVENTS / str(ano)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{evid}.ms"

    if dest.exists() and dest.stat().st_size > 0:
        with stats_lock:
            stats["ja_existia"] += 1
        return "ja_existia"

    key = f"event_waveforms/{ano}/{ano}_{doy:03d}/{evid}.ms"

    # 1. baixar em memória
    try:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, key, buf)
        bytes_brutos = buf.tell()
        buf.seek(0)
    except Exception as e:
        log.warning(f"  [{evid}] erro download: {e}")
        with stats_lock:
            stats["erro_download"] += 1
        return "erro_download"

    # 2. parsear
    try:
        st = obspy.read(buf)
    except Exception as e:
        log.warning(f"  [{evid}] erro parse: {e}")
        with stats_lock:
            stats["erro_parse"] += 1
        return "erro_parse"

    # 3. filtrar BHZ + estação válida
    traces_filtrados = []
    for tr in st:
        if tr.stats.channel != CANAL_ALVO:
            continue
        chave = f"{tr.stats.network}.{tr.stats.station}"
        if chave not in estacoes_validas:
            continue
        traces_filtrados.append(tr)

    if not traces_filtrados:
        with stats_lock:
            stats["sem_bhz_valido"] += 1
        return "sem_bhz_valido"

    # 4. salvar enxuto
    st_out = obspy.Stream(traces=traces_filtrados)
    try:
        st_out.write(str(dest), format="MSEED")
        bytes_salvos = dest.stat().st_size
    except Exception as e:
        log.warning(f"  [{evid}] erro salvar: {e}")
        with stats_lock:
            stats["erro_parse"] += 1
        return "erro_parse"

    with stats_lock:
        stats["ok"] += 1
        stats["bytes_baixados"] += bytes_brutos
        stats["bytes_salvos"] += bytes_salvos

    return "ok"


def main():
    log.info("=" * 60)
    log.info("DOWNLOAD DE EVENTS — SCEDC (estrutura híbrida)")
    log.info(f"  RAW destino:  {DRIVE_BASE}")
    log.info(f"  LOGS destino: {DIR_LOGS}")
    log.info(f"  Log file:     {log_file}")
    log.info("=" * 60)

    # checagem prévia
    if not DRIVE_BASE.exists():
        log.error(f"Drive não montado ou pasta não existe: {DRIVE_BASE}")
        log.error("Verifique se o Google Drive for Desktop está rodando.")
        sys.exit(1)

    if not CSV_EVENTOS.exists():
        log.error(f"CSV de eventos não encontrado: {CSV_EVENTOS}")
        log.error("Copie de C:\\TCC_data\\catalog\\eventos_alvo.csv pro Drive antes.")
        sys.exit(1)

    estacoes = carregar_estacoes_validas()
    log.info(f"Estações válidas (com XML): {len(estacoes)}")
    if len(estacoes) == 0:
        log.error(f"Nenhum XML em {DIR_XML}. Copie os XMLs pro Drive antes.")
        sys.exit(1)

    eventos = carregar_eventos()
    log.info(f"Eventos no catálogo: {len(eventos)}")

    DIR_EVENTS.mkdir(parents=True, exist_ok=True)
    s3 = fazer_cliente_s3()

    log.info(f"Iniciando download com {N_THREADS} threads...")
    log.info("(Ctrl+C interrompe; rodar de novo retoma de onde parou)")
    log.info("-" * 60)

    t0 = time.time()
    n_total = len(eventos)
    n_processados = 0

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futures = {ex.submit(baixar_e_filtrar_evento, s3, ev, estacoes): ev for ev in eventos}
        for fut in as_completed(futures):
            n_processados += 1
            try:
                fut.result()
            except Exception as e:
                log.error(f"  exceção não tratada: {e}")

            if n_processados % 25 == 0 or n_processados == n_total:
                elapsed = time.time() - t0
                rate = n_processados / elapsed if elapsed > 0 else 0
                eta = (n_total - n_processados) / rate if rate > 0 else 0
                log.info(
                    f"  {n_processados}/{n_total} "
                    f"({100*n_processados/n_total:.1f}%) "
                    f"| ok={stats['ok']} "
                    f"existia={stats['ja_existia']} "
                    f"sem_bhz={stats['sem_bhz_valido']} "
                    f"err_dl={stats['erro_download']} "
                    f"err_parse={stats['erro_parse']} "
                    f"| {rate:.1f} ev/s | ETA {eta/60:.1f}min"
                )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("DOWNLOAD CONCLUÍDO")
    log.info(f"Tempo total:      {elapsed/60:.1f} min")
    log.info(f"OK:               {stats['ok']}")
    log.info(f"Já existiam:      {stats['ja_existia']}")
    log.info(f"Sem BHZ válido:   {stats['sem_bhz_valido']}")
    log.info(f"Erro download:    {stats['erro_download']}")
    log.info(f"Erro parse:       {stats['erro_parse']}")
    log.info(f"Bytes baixados:   {stats['bytes_baixados']/1024/1024/1024:.2f} GB")
    log.info(f"Bytes salvos:     {stats['bytes_salvos']/1024/1024/1024:.2f} GB")
    if stats["bytes_baixados"] > 0:
        ratio = stats["bytes_salvos"] / stats["bytes_baixados"]
        log.info(f"Taxa de filtragem: {100*ratio:.1f}% (quanto menor, mais filtrou)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()