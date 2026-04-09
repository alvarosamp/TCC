r"""
Script 2 — Baixar catálogos SCEDC 2015–2018 e filtrar eventos-alvo.
- Baixa os 4 catálogos anuais do bucket s3://scedc-pds/
- Filtra eventos com magnitude >= 2.5
- Salva eventos_alvo.csv com: evid, ano, doy, datetime, mag, lat, lon, depth
- NÃO baixa waveforms ainda — só o catálogo (texto, leve).
"""

import logging
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("catalogo")

BASE_DIR = Path(r"C:\TCC_data")
CATALOG_DIR = BASE_DIR / "catalog"
BUCKET = "scedc-pds"
ANOS = [2015, 2016, 2017, 2018]
MAG_MIN = 2.5

s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED),
)


def baixar_catalogo_ano(ano: int) -> Path:
    key = f"earthquake_catalogs/SCEC_DC/{ano}.catalog"
    dest = CATALOG_DIR / f"{ano}.catalog"
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"  {ano}: já existe ({dest.stat().st_size/1024:.1f} KB), pulando download")
        return dest
    log.info(f"  {ano}: baixando s3://{BUCKET}/{key}")
    s3.download_file(BUCKET, key, str(dest))
    log.info(f"  {ano}: OK ({dest.stat().st_size/1024:.1f} KB)")
    return dest


def parsear_catalogo(path: Path, ano: int) -> list[dict]:
    """
    Formato SCEC_DC catalog (texto, colunas separadas por espaço).
    Linhas começando com '#' são cabeçalho/comentário.
    Colunas típicas (pode variar): YYYY/MM/DD HH:MM:SS.ss ET GT MAG M LAT LON DEPTH Q EVID NPH NGRMS
    A gente parseia defensivamente: pega data, mag, lat, lon, depth, evid.
    """
    eventos = []
    descartados_parse = 0
    descartados_mag = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            partes = linha.split()
            if len(partes) < 11:
                descartados_parse += 1
                continue
            try:
                data_str = partes[0]      # YYYY/MM/DD
                hora_str = partes[1]      # HH:MM:SS.ss
                mag = float(partes[4])    # MAG
                lat = float(partes[6])
                lon = float(partes[7])
                depth = float(partes[8])
                evid = partes[10]         # EVID
            except (ValueError, IndexError):
                descartados_parse += 1
                continue

            if mag < MAG_MIN:
                descartados_mag += 1
                continue

            # calcula DOY
            try:
                from datetime import datetime
                dt = datetime.strptime(f"{data_str} {hora_str[:8]}", "%Y/%m/%d %H:%M:%S")
                doy = dt.timetuple().tm_yday
            except ValueError:
                descartados_parse += 1
                continue

            eventos.append({
                "evid": evid,
                "ano": ano,
                "doy": doy,
                "datetime": dt.isoformat(),
                "mag": mag,
                "lat": lat,
                "lon": lon,
                "depth": depth,
            })

    log.info(f"  {ano}: {len(eventos)} eventos M>={MAG_MIN}  "
             f"(descartados: {descartados_mag} por mag, {descartados_parse} por parse)")
    return eventos


def main():
    log.info("=" * 60)
    log.info("BAIXAR CATÁLOGO — anos 2015–2018, M >= 2.5")
    log.info("=" * 60)

    log.info("Etapa 1/3 — Baixando catálogos anuais...")
    paths = {ano: baixar_catalogo_ano(ano) for ano in ANOS}

    log.info("Etapa 2/3 — Parseando e filtrando...")
    todos = []
    for ano, path in paths.items():
        todos.extend(parsear_catalogo(path, ano))

    log.info(f"TOTAL: {len(todos)} eventos M>={MAG_MIN} em {len(ANOS)} anos")

    if len(todos) == 0:
        log.error("Nenhum evento parseado. Formato do catálogo deve ser diferente do esperado.")
        log.error("Vou imprimir as 5 primeiras linhas não-comentário do catálogo de 2017:")
        with open(paths[2017], "r", encoding="utf-8", errors="replace") as f:
            n = 0
            for linha in f:
                if linha.strip() and not linha.startswith("#"):
                    log.error(f"  >>> {linha.rstrip()}")
                    n += 1
                    if n >= 5:
                        break
        sys.exit(1)

    log.info("Etapa 3/3 — Salvando eventos_alvo.csv...")
    import csv
    out = CATALOG_DIR / "eventos_alvo.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["evid", "ano", "doy", "datetime", "mag", "lat", "lon", "depth"])
        w.writeheader()
        w.writerows(todos)
    log.info(f"  OK  {out}  ({len(todos)} linhas)")

    # estatísticas rápidas
    log.info("-" * 60)
    log.info("Distribuição por ano:")
    from collections import Counter
    por_ano = Counter(e["ano"] for e in todos)
    for ano in sorted(por_ano):
        log.info(f"  {ano}: {por_ano[ano]} eventos")
    log.info("Distribuição por magnitude:")
    bins = [(2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 5.0), (5.0, 10.0)]
    for lo, hi in bins:
        n = sum(1 for e in todos if lo <= e["mag"] < hi)
        log.info(f"  M [{lo:.1f}, {hi:.1f}): {n}")

    log.info("=" * 60)
    log.info("CATÁLOGO PRONTO. Me manda a saída.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()