import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

BUCKET = "scedc-pds"

# Pasta raiz onde você quer salvar os dados
DATA_ROOT = Path(r"C:\Users\vish8\OneDrive\Documentos\SeriesTemporaisSismicas\data")

def make_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))

def list_all_keys(s3, prefix: str, max_keys: int | None = None) -> list[str]:
    keys: list[str] = []
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}

    while True:
        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            # ignora "pastas"
            if key.endswith("/"):
                continue
            keys.append(key)
            if max_keys is not None and len(keys) >= max_keys:
                return keys

        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    return keys

def download_key(s3, key: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(out_path))
    except ClientError as e:
        raise RuntimeError(f"Falha ao baixar {key}: {e}")

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # ✅ Comece pequeno: baixe metadados (leve) para provar o pipeline.
    # Depois você muda para "continuous_waveforms/" ou "event_waveforms/".
    prefix = "FDSNstationXML/"   # leve (metadados de estações)
    max_files = 50               # controla o tamanho do download (comece pequeno)

    s3 = make_s3_client()
    print(f"Listando keys em s3://{BUCKET}/{prefix} ...")
    keys = list_all_keys(s3, prefix=prefix, max_keys=max_files)
    print(f"Encontradas {len(keys)} keys (limit={max_files}).")

    # Salva inventário do que será baixado
    inventory_file = DATA_ROOT / "inventory_scedc_keys.txt"
    inventory_file.write_text("\n".join(keys), encoding="utf-8")
    print(f"Inventário salvo em: {inventory_file}")

    # Baixa mantendo a estrutura de pastas a partir do prefix
    print("Baixando arquivos...")
    for key in tqdm(keys):
        out_path = DATA_ROOT / "scedc-pds" / key  # espelha o bucket localmente
        if out_path.exists():
            continue
        download_key(s3, key, out_path)

    print("\n✅ Download concluído!")
    print(f"Arquivos em: {DATA_ROOT / 'scedc-pds'}")

if __name__ == "__main__":
    main()
