import argparse
import os
from pathlib import Path

import numpy as np


def default_data_dir():
    fallback = "D:/tCC/processed" if os.name == "nt" else "/mnt/d/tCC/processed"
    return Path(os.environ.get("TCC_PROCESSED_DIR", fallback))


def flatten_windows(x, expected_size):
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    elif x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Formato de X nao suportado: {x.shape}")

    if x.shape[1] != expected_size:
        raise ValueError(f"Janela tem {x.shape[1]} amostras, esperado {expected_size}")

    return x.astype(np.float32, copy=False)


def c_float(value):
    return f"{float(value):.9g}f"


def write_header(output_path, x, y, already_preprocessed):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_windows, window_size = x.shape

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#pragma once\n\n")
        f.write("// Gerado por scripts/export_balanced_dataset.py.\n")
        f.write("// Nao edite manualmente; regenere a partir do NPZ quando necessario.\n\n")

        flag = "true" if already_preprocessed else "false"
        f.write(f"constexpr bool kRealDatasetAlreadyPreprocessed = {flag};\n")
        f.write(f"constexpr int kRealDatasetWindowCount = {n_windows};\n")
        f.write(f"constexpr int kRealDatasetWindowSize = {window_size};\n\n")

        f.write(f"const int real_dataset_labels[{n_windows}] = {{")
        f.write(", ".join(str(int(v)) for v in y))
        f.write("};\n\n")

        f.write(f"const float real_dataset_windows[{n_windows}][{window_size}] = {{\n")
        for row in x:
            f.write("  {")
            f.write(", ".join(c_float(v) for v in row))
            f.write("},\n")
        f.write("};\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--dataset", default="dataset_v4_split_evento.npz")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("include/real_dataset.h"))
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use apenas se as janelas exportadas forem cruas e precisarem do preprocessing no ESP32.",
    )
    args = parser.parse_args()

    npz_path = args.data_dir / args.dataset
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {npz_path}")

    data = np.load(npz_path)

    x = flatten_windows(data[f"X_{args.split}"], args.window_size)
    y = np.asarray(data[f"y_{args.split}"]).astype(np.int32, copy=False)

    idx_normal = np.where(y == 0)[0]
    idx_anomalo = np.where(y == 1)[0]

    if len(idx_normal) < args.per_class:
        raise ValueError(f"Normais insuficientes: {len(idx_normal)}")
    if len(idx_anomalo) < args.per_class:
        raise ValueError(f"Anomalos insuficientes: {len(idx_anomalo)}")

    rng = np.random.default_rng(args.seed)

    chosen_normal = rng.choice(idx_normal, size=args.per_class, replace=False)
    chosen_anomalo = rng.choice(idx_anomalo, size=args.per_class, replace=False)

    chosen = np.concatenate([chosen_normal, chosen_anomalo])
    rng.shuffle(chosen)

    x_out = x[chosen]
    y_out = y[chosen]

    write_header(
        args.output,
        x_out,
        y_out,
        already_preprocessed=not args.raw,
    )

    print(f"OK: header gerado em {args.output}")
    print(f"Total: {len(y_out)}")
    print(f"Normais:  {(y_out == 0).sum()}")
    print(f"Anomalos: {(y_out == 1).sum()}")
    print(f"kRealDatasetAlreadyPreprocessed = {not args.raw}")


if __name__ == "__main__":
    main()