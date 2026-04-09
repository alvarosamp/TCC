"""
Script 1 — Setup e teste de acesso ao bucket SCEDC.
- Cria a estrutura de pastas em C:\TCC_data\
- Verifica se boto3 está instalado
- Testa acesso anônimo ao bucket público s3://scedc-pds/
- Lista alguns objetos pra confirmar que está tudo OK
- NÃO baixa nada pesado.
"""

import logging
import sys
from pathlib import Path

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup_s3")

# ---------- Configuração ----------
BASE_DIR = Path(r"C:\TCC_data")
SUBDIRS = [
    BASE_DIR / "raw" / "events",
    BASE_DIR / "raw" / "continuous",
    BASE_DIR / "raw" / "stationxml",
    BASE_DIR / "processed",
    BASE_DIR / "logs",
    BASE_DIR / "catalog",
]

BUCKET = "scedc-pds"
TEST_PREFIX = "event_waveforms/2017/2017_001/"  # dia 01/01/2017


def passo_1_checar_boto3():
    log.info("Passo 1/4 — Verificando boto3...")
    try:
        import boto3
        import botocore

        log.info(f"  boto3    = {boto3.__version__}")
        log.info(f"  botocore = {botocore.__version__}")
        return boto3, botocore
    except ImportError:
        log.error("boto3 NÃO está instalado no .venv.")
        log.error("Rode: pip install boto3")
        sys.exit(1)


def passo_2_criar_pastas():
    log.info("Passo 2/4 — Criando estrutura de pastas em C:\\TCC_data\\ ...")
    for d in SUBDIRS:
        d.mkdir(parents=True, exist_ok=True)
        log.info(f"  OK  {d}")


def passo_3_testar_acesso_anonimo(boto3, botocore):
    log.info("Passo 3/4 — Testando acesso anônimo ao bucket s3://scedc-pds/ ...")
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(signature_version=UNSIGNED),
    )

    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=TEST_PREFIX, MaxKeys=5)
    except botocore.exceptions.ClientError as e:
        log.error(f"Falhou ao listar bucket: {e}")
        sys.exit(1)
    except botocore.exceptions.EndpointConnectionError as e:
        log.error(f"Sem internet ou DNS travado: {e}")
        sys.exit(1)

    contents = resp.get("Contents", [])
    if not contents:
        log.warning("Bucket respondeu mas o prefixo de teste está vazio.")
        log.warning(f"  Prefixo testado: {TEST_PREFIX}")
        log.warning("  Não é fatal — pode ser que o dia escolhido não tenha eventos.")
    else:
        log.info(f"  Bucket respondeu OK. {len(contents)} objetos no prefixo de teste:")
        for obj in contents:
            size_mb = obj["Size"] / (1024 * 1024)
            log.info(f"    {obj['Key']}  ({size_mb:.2f} MB)")

    return s3


def passo_4_testar_catalogo(s3):
    log.info("Passo 4/4 — Verificando catálogo de eventos 2017 ...")
    catalog_key = "earthquake_catalogs/SCEC_DC/2017.catalog"
    try:
        head = s3.head_object(Bucket=BUCKET, Key=catalog_key)
        size_mb = head["ContentLength"] / (1024 * 1024)
        log.info(f"  OK  {catalog_key}  ({size_mb:.2f} MB)")
    except Exception as e:
        log.error(f"  Não achou o catálogo no caminho esperado: {e}")
        log.error("  Vou precisar ajustar o caminho do catálogo no próximo script.")


def main():
    log.info("=" * 60)
    log.info("SETUP S3 — TCC Álvaro")
    log.info("=" * 60)

    boto3, botocore = passo_1_checar_boto3()
    passo_2_criar_pastas()
    s3 = passo_3_testar_acesso_anonimo(boto3, botocore)
    passo_4_testar_catalogo(s3)

    log.info("=" * 60)
    log.info("SETUP CONCLUÍDO. Se tudo acima estiver OK, me avise.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()