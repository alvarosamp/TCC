from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import processed_dir, windows_dir


@dataclass(frozen=True)
class SplitData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _make_labels(n_normal: int, n_event: int) -> np.ndarray:
    return np.concatenate(
        [np.zeros(n_normal, dtype=np.int64), np.ones(n_event, dtype=np.int64)]
    )


def _stack_normal_event(normal: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([normal, event], axis=0)
    y = _make_labels(len(normal), len(event))
    return _as_float32(X), y


def load_legacy_windows(version: str = "v2") -> SplitData:
    """Load the repository artefacts: train has normal windows only.

    This matches artefacts/data/windows/windows_noise_v2.npz and
    windows_events_v2.npz. The validation set is used to choose thresholds;
    the test set is used only for final reporting.
    """

    suffix = "_v2" if version == "v2" else ""
    wdir = windows_dir()
    noise = np.load(wdir / f"windows_noise{suffix}.npz")
    events = np.load(wdir / f"windows_events{suffix}.npz")

    X_train = _as_float32(noise["X_train"])
    y_train = np.zeros(len(X_train), dtype=np.int64)
    X_val, y_val = _stack_normal_event(noise["X_val"], events["X_val"])
    X_test, y_test = _stack_normal_event(noise["X_test"], events["X_test"])

    return SplitData(
        name=f"legacy_{version}",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def load_npz_split(path: Path, name: str | None = None) -> SplitData:
    data = np.load(path)
    required = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")

    return SplitData(
        name=name or path.stem,
        X_train=_as_float32(data["X_train"]),
        y_train=np.asarray(data["y_train"], dtype=np.int64),
        X_val=_as_float32(data["X_val"]),
        y_val=np.asarray(data["y_val"], dtype=np.int64),
        X_test=_as_float32(data["X_test"]),
        y_test=np.asarray(data["y_test"], dtype=np.int64),
    )


def load_available_splits(dataset: str = "auto") -> list[SplitData]:
    """Load available data with a stable priority.

    dataset options:
    - auto: use dataset_v3_split_*.npz if TCC_PROCESSED_DIR exists, else legacy v2.
    - legacy_v2: use artefacts/data/windows v2 files.
    - legacy_v1: use artefacts/data/windows v1 files.
    - v3: require dataset_v3_split_estacao.npz and dataset_v3_split_temporal.npz.
    """

    if dataset == "legacy_v2":
        return [load_legacy_windows("v2")]
    if dataset == "legacy_v1":
        return [load_legacy_windows("v1")]

    pdir = processed_dir()
    if dataset in {"auto", "v3"} and pdir is not None:
        paths = [
            (pdir / "dataset_v3_split_estacao.npz", "estacao"),
            (pdir / "dataset_v3_split_temporal.npz", "temporal"),
        ]
        if all(path.exists() for path, _ in paths):
            return [load_npz_split(path, name) for path, name in paths]
        if dataset == "v3":
            raise FileNotFoundError(
                "dataset_v3_split_estacao.npz and dataset_v3_split_temporal.npz "
                f"were not found in {pdir}"
            )

    return [load_legacy_windows("v2")]


def normal_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 0]


def event_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1]

