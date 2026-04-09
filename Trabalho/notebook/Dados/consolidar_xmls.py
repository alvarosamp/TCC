r"""
Script 2.5 — Consolida todos os XMLs encontrados em C:\TCC_data\raw\stationxml\
Lê de duas raízes (OneDrive e GitHub) e copia para um único destino.
Renomeia para o padrão {NET}.{STA}.xml para garantir consistência.
Em caso de duplicata, mantém o maior arquivo (assumindo mais completo).
"""

import logging
import shutil
from pathlib import Path

import obspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xmls")

DESTINO = Path(r"C:\TCC_data\raw\stationxml")
RAIZES = [
    Path(r"C:\Users\vish8\OneDrive\Documentos\TCC\data\scedc-pds\FDSNstationXML"),
    Path(r"C:\Users\vish8\OneDrive\Documentos\GitHub\TCC\data\scedc-pds\FDSNstationXML"),
]


def main():
    log.info("=" * 60)
    log.info("CONSOLIDAR XMLs")
    log.info("=" * 60)

    DESTINO.mkdir(parents=True, exist_ok=True)

    encontrados = []
    for raiz in RAIZES:
        if not raiz.exists():
            log.warning(f"Raiz não existe (pulando): {raiz}")
            continue
        log.info(f"Varrendo: {raiz}")
        encontrados.extend(raiz.rglob("*.xml"))
    log.info(f"Total de XMLs encontrados: {len(encontrados)}")

    copiados = 0
    pulados_invalidos = 0
    pulados_duplicados = 0

    for src in encontrados:
        try:
            inv = obspy.read_inventory(str(src))
        except Exception as e:
            log.warning(f"  inválido (não é StationXML): {src.name} — {e}")
            pulados_invalidos += 1
            continue

        # extrai NET.STA do inventário
        try:
            net = inv[0].code
            sta = inv[0][0].code
        except (IndexError, AttributeError):
            log.warning(f"  sem net/sta: {src.name}")
            pulados_invalidos += 1
            continue

        nome_padrao = f"{net}.{sta}.xml"
        dest = DESTINO / nome_padrao

        if dest.exists():
            if src.stat().st_size > dest.stat().st_size:
                log.info(f"  duplicata maior, sobrescrevendo: {nome_padrao}")
                shutil.copy2(src, dest)
            else:
                pulados_duplicados += 1
            continue

        shutil.copy2(src, dest)
        copiados += 1

    log.info("-" * 60)
    log.info(f"Copiados:    {copiados}")
    log.info(f"Duplicados:  {pulados_duplicados}")
    log.info(f"Inválidos:   {pulados_invalidos}")
    log.info(f"Total final em {DESTINO}: {len(list(DESTINO.glob('*.xml')))}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()