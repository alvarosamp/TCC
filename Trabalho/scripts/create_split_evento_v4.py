from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validar_arquivos(data_dir: Path, split_base: str) -> tuple[Path, Path]:
    inventario_path = data_dir / "inventario_v3.csv"
    dataset_path = data_dir / f"dataset_v3_split_{split_base}.npz"
    missing = [p.name for p in [inventario_path, dataset_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em {data_dir}: {', '.join(missing)}"
        )
    return inventario_path, dataset_path


def reconstruir_dataset_total(data_dir: Path, split_base: str):
    inventario_path, dataset_path = validar_arquivos(data_dir, split_base)
    df = pd.read_csv(inventario_path)
    data = np.load(dataset_path)
    split_col = f"split_{split_base}"

    required_cols = {"tipo", "evid", split_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes no inventario: {sorted(missing_cols)}")

    required_keys = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise ValueError(f"Chaves ausentes no NPZ: {sorted(missing_keys)}")

    n_total = len(df)
    sample_shape = data["X_train"].shape[1:]
    X_all = np.empty((n_total, *sample_shape), dtype=data["X_train"].dtype)
    y_all = np.empty(n_total, dtype=data["y_train"].dtype)

    for part in ["train", "val", "test"]:
        idx = df.index[df[split_col] == part].to_numpy()
        X_part = data[f"X_{part}"]
        y_part = data[f"y_{part}"]
        if len(idx) != len(X_part):
            raise ValueError(
                f"Tamanho inconsistente em {part}: "
                f"inventario={len(idx)} npz={len(X_part)}"
            )
        X_all[idx] = X_part
        y_all[idx] = y_part

    y_inventory = (df["tipo"] == "anomalo").astype(y_all.dtype).to_numpy()
    if not np.array_equal(y_all, y_inventory):
        raise ValueError("Labels reconstruidos nao batem com inventario_v3.csv")
    return X_all, y_all, df


def dividir_ids(ids, rng, train_ratio: float, val_ratio: float):
    ids = np.array(sorted(ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        set(ids[:n_train]),
        set(ids[n_train:n_train + n_val]),
        set(ids[n_train + n_val:]),
    )


def criar_split_evento(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    splits = np.full(len(df), fill_value="", dtype=object)

    mask_anomalo = df["tipo"] == "anomalo"
    event_ids = df.loc[mask_anomalo, "evid"].dropna().unique()
    ev_train, ev_val, ev_test = dividir_ids(event_ids, rng, train_ratio, val_ratio)

    for idx, evid in df.loc[mask_anomalo, "evid"].items():
        if evid in ev_train:
            splits[idx] = "train"
        elif evid in ev_val:
            splits[idx] = "val"
        elif evid in ev_test:
            splits[idx] = "test"
        else:
            raise ValueError(f"Evento anomalo sem split definido: {evid}")

    normal_idx = df.index[df["tipo"] == "normal"].to_numpy()
    rng.shuffle(normal_idx)
    n_train = int(len(normal_idx) * train_ratio)
    n_val = int(len(normal_idx) * val_ratio)
    splits[normal_idx[:n_train]] = "train"
    splits[normal_idx[n_train:n_train + n_val]] = "val"
    splits[normal_idx[n_train + n_val:]] = "test"

    if np.any(splits == ""):
        raise ValueError(f"{int(np.sum(splits == ''))} linhas sem split")
    return splits


def salvar_split(data_dir: Path, nome: str, X_all, y_all, df: pd.DataFrame, splits, seed: int):
    payload = {}
    for part in ["train", "val", "test"]:
        mask = splits == part
        payload[f"X_{part}"] = X_all[mask]
        payload[f"y_{part}"] = y_all[mask]

    out_npz = data_dir / f"dataset_v4_split_{nome}.npz"
    out_csv = data_dir / f"inventario_v4_split_{nome}.csv"
    out_json = data_dir / f"dataset_v4_split_{nome}_info.json"
    np.savez_compressed(out_npz, **payload)

    df_out = df.copy()
    df_out[f"split_{nome}"] = splits
    df_out.to_csv(out_csv, index=False)

    counts = {}
    for part in ["train", "val", "test"]:
        y = payload[f"y_{part}"]
        total = int(len(y))
        anomaly = int((y == 1).sum())
        counts[part] = {
            "total": total,
            "normal": int((y == 0).sum()),
            "anomalo": anomaly,
            "baseline_auc_pr": anomaly / total if total else 0.0,
        }

    leakage = (
        df_out[df_out["tipo"] == "anomalo"]
        .groupby("evid")[f"split_{nome}"]
        .nunique()
    )
    info = {
        "nome": nome,
        "seed": seed,
        "data_dir": str(data_dir),
        "arquivos": {
            "dataset": str(out_npz),
            "inventario": str(out_csv),
            "info": str(out_json),
        },
        "counts": counts,
        "n_eventos_vazando": int((leakage > 1).sum()),
    }
    out_json.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria dataset v4 com split por evento.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-base", default="temporal")
    parser.add_argument("--nome", default="evento")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_all, y_all, df = reconstruir_dataset_total(args.data_dir, args.split_base)
    splits = criar_split_evento(df, seed=args.seed)
    info = salvar_split(args.data_dir, args.nome, X_all, y_all, df, splits, args.seed)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
