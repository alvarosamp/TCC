r"""
Script 4 — Baixar continuous waveforms do SCEDC.
Fonte de dados garantidamente NORMAL para o pipeline genérico de detecção
de anomalias. Não usa nenhuma informação sismológica — apenas "este trecho
foi gravado em horário sem evento catalogado".

Estrutura no S3:
  s3://scedc-pds/continuous_waveforms/{YYYY}/{YYYY_DOY}/{NET}{STA}_{LOC}{CHA}_{YYYY_DOY}.ms

Salva em:
  G:\Meu Drive\TCC\data2\raw\continuous\{YYYY_DOY}\{NET}.{STA}.BHZ.{YYYY_DOY}.ms
"""

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

# silencia warnings cosméticos
logging.getLogger("botocore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ---------- Config ----------
DRIVE_BASE = Path(r"G:\Meu Drive\TCC\data2")
LOCAL_BASE = Path(r"C:\TCC_data")

DIR_CONT = DRIVE_BASE / "raw" / "continuous"
DIR_LOGS = LOCAL_BASE / "logs"

BUCKET = "scedc-pds"
N_THREADS = 6

# Estações escolhidas: aparecem em todos os events inspecionados
ESTACOES = [
    ("CI", "PASC"),
    ("AZ", "BZN"),
    ("CI", "ADO"),
    ("CI", "ALP"),
    ("CI", "ARV"),
]

# Dias espalhados em 2017 (~mensalmente, mistura semana/fim de semana)
ANO = 2017
DIAS_DOY = [45, 105, 165, 195, 225, 255, 315]

CANAL = "BHZ"

# ---------- Logging ----------
DIR_LOGS.mkdir(parents=True, exist_ok=True)
log_file = DIR_LOGS / f"download_continuous_{int(time.time())}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dl_cont")

stats = {"ok": 0, "ja_existia": 0, "sem_canal": 0, "erro_dl": 0, "erro_parse": 0,
         "bytes_baixados": 0, "bytes_salvos": 0}
stats_lock = Lock()


def fazer_cliente_s3():
    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=N_THREADS * 4,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def listar_keys_estacao_dia(s3, net, sta, ano, doy):
    """Lista todas as chaves S3 para essa estação nesse dia. SCEDC tem múltiplos
    location codes (00, 10) — pega a primeira BHZ encontrada."""
    prefix = f"continuous_waveforms/{ano}/{ano}_{doy:03d}/"
    # padrão de nome no SCEDC: {NET}{STA}__{LOC}{CHA}___{YYYY_DOY}.ms
    # mas pode variar, então listamos e filtramos
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.rsplit("/", 1)[-1]
            # filtra por NET, STA e canal BHZ
            if net in fname and sta in fname and CANAL in fname:
                keys.append(key)
    return keys


def baixar_um(s3, net, sta, doy):
    """Baixa um (estação, dia). Retorna status string."""
    dest_dir = DIR_CONT / f"{ANO}_{doy:03d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{net}.{sta}.{CANAL}.{ANO}_{doy:03d}.ms"

    if dest.exists() and dest.stat().st_size > 1024:
        with stats_lock:
            stats["ja_existia"] += 1
        return "ja_existia"

    try:
        keys = listar_keys_estacao_dia(s3, net, sta, ANO, doy)
    except Exception as e:
        log.warning(f"  [{net}.{sta} {doy:03d}] erro list: {e}")
        with stats_lock:
            stats["erro_dl"] += 1
        return "erro_dl"

    if not keys:
        log.warning(f"  [{net}.{sta} {doy:03d}] nenhum BHZ disponível")
        with stats_lock:
            stats["sem_canal"] += 1
        return "sem_canal"

    # se houver mais de uma (location codes diferentes), pega a primeira
    key = sorted(keys)[0]

    try:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, key, buf)
        bytes_brutos = buf.tell()
        buf.seek(0)
    except Exception as e:
        log.warning(f"  [{net}.{sta} {doy:03d}] erro download: {e}")
        with stats_lock:
            stats["erro_dl"] += 1
        return "erro_dl"

    try:
        st = obspy.read(buf)
    except Exception as e:
        log.warning(f"  [{net}.{sta} {doy:03d}] erro parse: {e}")
        with stats_lock:
            stats["erro_parse"] += 1
        return "erro_parse"

    # filtra só BHZ (deveria já ser, mas garante)
    st_bhz = obspy.Stream([tr for tr in st if tr.stats.channel == CANAL])
    if len(st_bhz) == 0:
        with stats_lock:
            stats["sem_canal"] += 1
        return "sem_canal"

    try:
        st_bhz.write(str(dest), format="MSEED")
        bytes_salvos = dest.stat().st_size
    except Exception as e:
        log.warning(f"  [{net}.{sta} {doy:03d}] erro salvar: {e}")
        with stats_lock:
            stats["erro_parse"] += 1
        return "erro_parse"

    with stats_lock:
        stats["ok"] += 1
        stats["bytes_baixados"] += bytes_brutos
        stats["bytes_salvos"] += bytes_salvos

    log.info(f"  [{net}.{sta} {doy:03d}] OK ({bytes_salvos/1024/1024:.1f} MB)")
    return "ok"


def main():
    log.info("=" * 70)
    log.info("DOWNLOAD CONTINUOUS WAVEFORMS — fonte de dados NORMAIS")
    log.info(f"  Estações: {[f'{n}.{s}' for n,s in ESTACOES]}")
    log.info(f"  Dias DOY: {DIAS_DOY} de {ANO}")
    log.info(f"  Total:    {len(ESTACOES) * len(DIAS_DOY)} arquivos")
    log.info(f"  Destino:  {DIR_CONT}")
    log.info(f"  Log:      {log_file}")
    log.info("=" * 70)

    if not DRIVE_BASE.exists():
        log.error(f"Drive não montado: {DRIVE_BASE}")
        sys.exit(1)

    DIR_CONT.mkdir(parents=True, exist_ok=True)
    s3 = fazer_cliente_s3()

    tarefas = [(net, sta, doy) for (net, sta) in ESTACOES for doy in DIAS_DOY]
    n_total = len(tarefas)
    log.info(f"Iniciando {n_total} downloads com {N_THREADS} threads...")
    log.info("-" * 70)

    t0 = time.time()
    n_proc = 0

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futs = {ex.submit(baixar_um, s3, n, s, d): (n, s, d) for (n, s, d) in tarefas}
        for fut in as_completed(futs):
            n_proc += 1
            try:
                fut.result()
            except Exception as e:
                log.error(f"  exceção: {e}")

            if n_proc % 5 == 0 or n_proc == n_total:
                elapsed = time.time() - t0
                rate = n_proc / elapsed if elapsed > 0 else 0
                eta = (n_total - n_proc) / rate if rate > 0 else 0
                log.info(
                    f"  {n_proc}/{n_total} ({100*n_proc/n_total:.1f}%) "
                    f"| ok={stats['ok']} existia={stats['ja_existia']} "
                    f"sem_canal={stats['sem_canal']} err={stats['erro_dl']+stats['erro_parse']} "
                    f"| ETA {eta/60:.1f}min"
                )

    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info("DOWNLOAD CONTÍNUO CONCLUÍDO")
    log.info(f"  Tempo total:    {elapsed/60:.1f} min")
    log.info(f"  OK:             {stats['ok']}")
    log.info(f"  Já existiam:    {stats['ja_existia']}")
    log.info(f"  Sem canal BHZ:  {stats['sem_canal']}")
    log.info(f"  Erro download:  {stats['erro_dl']}")
    log.info(f"  Erro parse:     {stats['erro_parse']}")
    log.info(f"  Bytes baixados: {stats['bytes_baixados']/1024/1024:.1f} MB")
    log.info(f"  Bytes salvos:   {stats['bytes_salvos']/1024/1024:.1f} MB")
    log.info("=" * 70)


if __name__ == "__main__":
    main()