#!/usr/bin/env python3
"""
GPU benchmarking runner for GGNN.
Consumes the first line of gpu-runs-ggnn.txt (CSV: dim,points,k) and appends results
to gpu_ggnn_performance.csv (one line per run), then removes the processed run.
"""

import csv
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

SEED = 42
RUNS_FILE = "gpu-runs-ggnn.txt"
OUTPUT_CSV = "gpu_ggnn_performance.csv"

# Data paths (match vector-analysis.ipynb/data-analysis.py)
DATA_DIR = os.environ.get("CLIC_DATA_DIR", "/workspace/data")
ARRAY_RECORD_GLOB = os.environ.get(
    "CLIC_ARRAY_RECORD_GLOB", "clic_edm_qq_pf-test.array_record-*"
)
FEATURE_KEY = os.environ.get("CLIC_FEATURE_KEY", "X")
FEATURE_WIDTH = int(os.environ.get("CLIC_FEATURE_WIDTH", "17"))

_FULL_DATA_CACHE: Optional[np.ndarray] = None

# GGNN parameters (mirroring generate_ggnn_performance.py defaults)
K_BUILD = 24
TAU_BUILD = 0.5
TAU_QUERY = 0.64
MAX_ITERATIONS = 400


def _empty_data(dimensions: int) -> np.ndarray:
    return np.empty((0, dimensions), dtype=np.float32)


def _load_all_from_array_record(data_dir: str, dimensions: int) -> np.ndarray:
    try:
        import tensorflow as tf
        from array_record.python.array_record_data_source import ArrayRecordDataSource
    except ImportError as exc:
        raise ImportError(
            "tensorflow and array_record are required to load ArrayRecord data"
        ) from exc

    shard_paths = sorted(Path(data_dir).glob(ARRAY_RECORD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(
            f"No ArrayRecord shards found in {data_dir} with glob {ARRAY_RECORD_GLOB}"
        )

    def decode_example(raw_bytes: bytes) -> np.ndarray:
        example = tf.train.Example()
        example.ParseFromString(raw_bytes)
        values = example.features.feature[FEATURE_KEY].float_list.value
        if not values:
            return np.empty((0, dimensions), dtype=np.float32)
        x = np.asarray(values, dtype=np.float32)
        if x.size % FEATURE_WIDTH != 0:
            raise ValueError(
                f"Feature '{FEATURE_KEY}' size {x.size} is not divisible by {FEATURE_WIDTH}"
            )
        x = x.reshape(-1, FEATURE_WIDTH)[:, :dimensions]
        return x

    print("Counting total particles...")
    total_particles = 0
    for shard_path in shard_paths:
        with ArrayRecordDataSource(str(shard_path)) as ds:
            for i in range(len(ds)):
                x = decode_example(ds[i])
                total_particles += len(x)

    print(f"Loading {total_particles:,} particles...")
    all_particles = np.zeros((total_particles, dimensions), dtype=np.float32)
    particle_idx = 0

    for shard_path in shard_paths:
        with ArrayRecordDataSource(str(shard_path)) as ds:
            for i in range(len(ds)):
                x = decode_example(ds[i])
                if len(x) == 0:
                    continue
                all_particles[particle_idx:particle_idx + len(x)] = x
                particle_idx += len(x)

    return all_particles[:particle_idx]


def load_data(n_points: int, dimensions: int) -> np.ndarray:
    global _FULL_DATA_CACHE
    if n_points == 0:
        return _empty_data(dimensions)
    if _FULL_DATA_CACHE is None:
        print(f"Loading ArrayRecord data from {DATA_DIR}")
        _FULL_DATA_CACHE = _load_all_from_array_record(DATA_DIR, dimensions)
    if n_points >= len(_FULL_DATA_CACHE):
        if n_points > len(_FULL_DATA_CACHE):
            print(
                f"Requested {n_points} points, using all {len(_FULL_DATA_CACHE)} available."
            )
        return _FULL_DATA_CACHE
    return _FULL_DATA_CACHE[:n_points]


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def time_ggnn(data_gpu: torch.Tensor, k: int) -> Tuple[Optional[float], str]:
    if data_gpu.shape[0] == 0:
        return 0.0, "empty"
    try:
        import ggnn
    except ImportError:
        return None, "not_installed"

    try:
        ggnn.set_log_level(1)
        start = time.perf_counter()
        my_ggnn = ggnn.GGNN()
        my_ggnn.set_base(data_gpu)
        my_ggnn.set_return_results_on_gpu(True)
        my_ggnn.build(
            k_build=K_BUILD, tau_build=TAU_BUILD, measure=ggnn.DistanceMeasure.Euclidean
        )
        my_ggnn.query(
            data_gpu, k, TAU_QUERY, MAX_ITERATIONS, ggnn.DistanceMeasure.Euclidean
        )
        torch.cuda.synchronize()
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"GGNN error: {exc}")
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
        raise RuntimeError("No runs left in gpu-runs-ggnn.txt")
    return parse_run_line(lines[idx]), idx


def delete_run_line(path: str, line_index: int) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return
    remaining = lines[:line_index] + lines[line_index + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(remaining)


def append_result(
    path: str,
    dim: int,
    points: int,
    k: int,
    time_ms: Optional[float],
    status: str,
) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["dim", "points", "k", "time_ms", "status"])
        writer.writerow([dim, points, k, time_ms, status])


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(base_dir, RUNS_FILE)
    output_path = os.path.join(base_dir, OUTPUT_CSV)

    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"{RUNS_FILE} not found in {base_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU benchmarking")

    while True:
        try:
            (dim, points, k), line_index = read_next_run(runs_path)
        except RuntimeError as exc:
            print(str(exc))
            return

        print(f"Running GGNN benchmark: dim={dim}, points={points}, k={k}")
        data_np = load_data(points, dim)
        data_gpu = torch.tensor(data_np, dtype=torch.float32, device="cuda")
        time_ms, status = time_ggnn(data_gpu, k)
        append_result(output_path, dim, points, k, time_ms, status)
        delete_run_line(runs_path, line_index)
        print(f"Wrote results to {output_path} and removed run line.")


if __name__ == "__main__":
    main()
