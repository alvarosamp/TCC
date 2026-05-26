import argparse
import os
from pathlib import Path

import numpy as np


def default_data_dir():
    fallback = "D:/tCC/processed" if os.name == "nt" else "/mnt/d/tCC/processed"
    return Path(os.environ.get("TCC_PROCESSED_DIR", fallback))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exporta janelas reais de um NPZ do TCC para include/real_dataset.h."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--dataset", default="dataset_v4_split_evento.npz")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--classe", choices=["all", "normal", "anomalo"], default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--output", type=Path, default=Path("include/real_dataset.h"))
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use se as janelas exportadas ainda precisarem de z-score no ESP32.",
    )
    return parser.parse_args()


def selected_splits(split):
    return ["train", "val", "test"] if split == "all" else [split]


def flatten_windows(x, expected_size):
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[-1] == 1:
      x = x[:, :, 0]
    elif x.ndim == 3 and x.shape[1] == 1:
      x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Formato de X nao suportado: {x.shape}")
    if x.shape[1] != expected_size:
        raise ValueError(
            f"Janela com tamanho {x.shape[1]}, esperado {expected_size}."
        )

    return x.astype(np.float32, copy=False)


def load_split_arrays(npz_path, split, expected_size):
    data = np.load(npz_path)

    xs = []
    ys = []
    for part in selected_splits(split):
        x_key = f"X_{part}"
        y_key = f"y_{part}"
        if x_key not in data or y_key not in data:
            raise KeyError(f"Chaves ausentes no NPZ: {x_key}/{y_key}")
        xs.append(flatten_windows(data[x_key], expected_size))
        ys.append(np.asarray(data[y_key]).astype(np.int32, copy=False))

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def filter_class(x, y, classe):
    if classe == "all":
        return x, y
    target = 0 if classe == "normal" else 1
    mask = y == target
    return x[mask], y[mask]


def c_float(value):
    return f"{float(value):.9g}f"


def write_header(output_path, x, y, already_preprocessed):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_windows, window_size = x.shape

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#pragma once\n\n")
        f.write("// Gerado por scripts/export_real_dataset.py.\n")
        f.write("// Nao edite manualmente; regenere a partir do NPZ quando necessario.\n\n")
        f.write(f"constexpr bool kRealDatasetAlreadyPreprocessed = {'true' if already_preprocessed else 'false'};\n")
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
    args = parse_args()
    npz_path = args.data_dir / args.dataset

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {npz_path}")

    x, y = load_split_arrays(npz_path, args.split, args.window_size)
    x, y = filter_class(x, y, args.classe)

    end = args.start + args.limit
    x = x[args.start:end]
    y = y[args.start:end]

    if len(x) == 0:
        raise ValueError("Nenhuma janela selecionada para exportar.")

    write_header(args.output, x, y, already_preprocessed=not args.raw)

    print(f"OK: {len(x)} janelas exportadas para {args.output}")
    print(f"Labels: normal={(y == 0).sum()} anomalo={(y == 1).sum()}")
    print(f"Preprocessamento no ESP32: {'sim' if args.raw else 'nao'}")


if __name__ == "__main__":
    main()
