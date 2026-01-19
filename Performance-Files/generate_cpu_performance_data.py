#!/usr/bin/env python3
"""
CPU benchmarking runner for ScaNN, HNSWlib, Annoy, FAISS (CPU), and FGC (CPU).
Consumes the first line of cpu-runs.txt (CSV: dim,points,k) and appends results
to cpu_performance.csv (one line per method), then removes the processed run.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from fastgraphcompute import binned_select_knn


SEED = 42
RUNS_FILE = "cpu-runs.txt"
OUTPUT_CSV = "cpu_performance.csv"

# Data paths (match vector-analysis.ipynb/data-analysis.py)
DATA_DIR = os.environ.get("CLIC_DATA_DIR", "/workspace/data")
HDF5_PATH = os.environ.get("CLIC_HDF5_PATH", os.path.join(
    DATA_DIR, "clic_test_particles.h5"))
HDF5_DATASET = os.environ.get("CLIC_HDF5_DATASET", "particles")
ARRAY_RECORD_GLOB = os.environ.get(
    "CLIC_ARRAY_RECORD_GLOB", "clic_edm_qq_pf-test.array_record-*"
)
FEATURE_KEY = os.environ.get("CLIC_FEATURE_KEY", "X")
FEATURE_WIDTH = int(os.environ.get("CLIC_FEATURE_WIDTH", "17"))


def _empty_data(dimensions: int) -> np.ndarray:
    return np.empty((0, dimensions), dtype=np.float32)


def _sample_from_hdf5(
    path: str, points: int, dimensions: int, seed_offset: int = 0
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load HDF5 data") from exc

    if points == 0:
        return _empty_data(dimensions)

    with h5py.File(path, "r") as f:
        if HDF5_DATASET not in f:
            raise KeyError(f"Dataset '{HDF5_DATASET}' not found in {path}")
        dataset = f[HDF5_DATASET]
        total_points, total_dim = dataset.shape
        if dimensions > total_dim:
            raise ValueError(
                f"Requested dim={dimensions} exceeds dataset dim={total_dim}"
            )
        rng = np.random.default_rng(SEED + seed_offset)
        if points >= total_points:
            if points > total_points:
                print(
                    f"Requested {points} points, using all {total_points} available."
                )
            data = dataset[:, :dimensions]
        else:
            indices = rng.choice(total_points, size=points, replace=False)
            data = dataset[indices, :dimensions]
    return np.asarray(data, dtype=np.float32)


def _sample_from_array_record(
    data_dir: str, points: int, dimensions: int, seed_offset: int = 0
) -> np.ndarray:
    try:
        import tensorflow as tf
        from array_record.python.array_record_data_source import ArrayRecordDataSource
    except ImportError as exc:
        raise ImportError(
            "tensorflow and array_record are required to load ArrayRecord data"
        ) from exc

    if points == 0:
        return _empty_data(dimensions)

    shard_paths = sorted(Path(data_dir).glob(ARRAY_RECORD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(
            f"No ArrayRecord shards found in {data_dir} with glob {ARRAY_RECORD_GLOB}"
        )

    rng = np.random.default_rng(SEED + seed_offset)
    sample = np.empty((points, dimensions), dtype=np.float32)
    filled = 0
    seen = 0

    for shard_path in shard_paths:
        with ArrayRecordDataSource(str(shard_path)) as ds:
            for i in range(len(ds)):
                raw_bytes = ds[i]
                example = tf.train.Example()
                example.ParseFromString(raw_bytes)
                values = example.features.feature[FEATURE_KEY].float_list.value
                if not values:
                    continue
                x = np.asarray(values, dtype=np.float32)
                if x.size % FEATURE_WIDTH != 0:
                    raise ValueError(
                        f"Feature '{FEATURE_KEY}' size {x.size} is not divisible by {FEATURE_WIDTH}"
                    )
                x = x.reshape(-1, FEATURE_WIDTH)[:, :dimensions]
                for row in x:
                    seen += 1
                    if filled < points:
                        sample[filled] = row
                        filled += 1
                    else:
                        j = rng.integers(0, seen)
                        if j < points:
                            sample[j] = row

    if filled < points:
        print(f"Only collected {filled} points from ArrayRecord data.")
        return sample[:filled]
    return sample


def load_data(n_points: int, dimensions: int, seed_offset: int = 0) -> np.ndarray:
    if n_points == 0:
        return _empty_data(dimensions)
    if os.path.exists(HDF5_PATH):
        print(f"Loading HDF5 data from {HDF5_PATH}")
        return _sample_from_hdf5(HDF5_PATH, n_points, dimensions, seed_offset)
    print(f"Loading ArrayRecord data from {DATA_DIR}")
    return _sample_from_array_record(DATA_DIR, n_points, dimensions, seed_offset)


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def time_annoy(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        from annoy import AnnoyIndex
    except ImportError:
        return None, "not_installed"

    try:
        start = time.perf_counter()
        index = AnnoyIndex(data.shape[1], "euclidean")
        for i in range(len(data)):
            index.add_item(i, data[i])
        index.build(10)
        for i in range(len(data)):
            index.get_nns_by_vector(data[i], k, include_distances=False)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"Annoy error: {exc}")
        return None, "error"


def time_hnswlib(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        import hnswlib
    except ImportError:
        return None, "not_installed"

    try:
        num_elements = data.shape[0]
        start = time.perf_counter()
        index = hnswlib.Index(space="l2", dim=data.shape[1])
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(data, np.arange(num_elements))
        index.set_ef(max(50, k + 10))
        index.knn_query(data, k=k)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"HNSWlib error: {exc}")
        return None, "error"


def time_faiss_cpu(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        import faiss
    except ImportError:
        return None, "not_installed"

    try:
        start = time.perf_counter()
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data)
        index.search(data, k)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"FAISS error: {exc}")
        return None, "error"


def time_scann(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        import scann
    except ImportError:
        return None, "not_installed"

    try:
        # ScaNN works better with normalized data.
        norms = np.linalg.norm(data, axis=1)
        normalized_data = data / norms[:, np.newaxis]
        start = time.perf_counter()
        searcher = (
            scann.scann_ops_pybind.builder(normalized_data, k, "dot_product")
            .tree(
                num_leaves=min(2000, len(data) // 10),
                num_leaves_to_search=min(100, len(data) // 50),
                training_sample_size=min(250_000, len(data)),
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(min(100, k * 2))
            .build()
        )
        searcher.search_batched(normalized_data)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"ScaNN error: {exc}")
        return None, "error"


def time_fgc_cpu(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        start = time.perf_counter()
        coordinates = torch.tensor(
            data, dtype=torch.float32, device="cpu").contiguous()
        row_splits = torch.tensor(
            [0, len(data)], dtype=torch.int64, device="cpu")
        binned_select_knn(k, coordinates, row_splits,
                          direction=None, n_bins=None)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"FGC error: {exc}")
        return None, "error"


def parse_run_line(line: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid run line: {line}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def read_next_run(path: str) -> Tuple[Tuple[int, int, int], int]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise RuntimeError("No runs left in cpu-runs.txt")
    return parse_run_line(lines[idx]), idx


def delete_run_line(path: str, line_index: int) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return
    remaining = lines[:line_index] + lines[line_index + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(remaining)


def append_results(
    path: str,
    dim: int,
    points: int,
    k: int,
    results: Dict[str, Tuple[Optional[float], str]],
) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                ["dim", "points", "k", "method", "time_ms", "status"])
        for method, (time_ms, status) in results.items():
            writer.writerow([dim, points, k, method, time_ms, status])


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(base_dir, RUNS_FILE)
    output_path = os.path.join(base_dir, OUTPUT_CSV)

    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"{RUNS_FILE} not found in {base_dir}")

    try:
        (dim, points, k), line_index = read_next_run(runs_path)
    except RuntimeError as exc:
        print(str(exc))
        return

    print(f"Running CPU benchmarks: dim={dim}, points={points}, k={k}")
    data = load_data(points, dim)

    results: Dict[str, Tuple[Optional[float], str]] = {
        "scann": time_scann(data, k),
        "hnswlib": time_hnswlib(data, k),
        "annoy": time_annoy(data, k),
        "faiss_cpu": time_faiss_cpu(data, k),
        "fgc_cpu": time_fgc_cpu(data, k),
    }

    append_results(output_path, dim, points, k, results)
    delete_run_line(runs_path, line_index)
    print(f"Wrote results to {output_path} and removed run line.")


if __name__ == "__main__":
    main()
