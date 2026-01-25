#!/usr/bin/env python3
"""
GPU benchmarking runner for FGC (GPU).
Consumes the first line of gpu-runs-fgc_gpu.txt (CSV: dim,points,k) and appends results
to gpu_fgc_gpu_performance.csv (one line per run), then removes the processed run.
"""

import csv
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from fastgraphcompute import binned_select_knn

SEED = 42
RUNS_FILE = "gpu-runs-fgc_gpu.txt"
OUTPUT_CSV = "gpu_fgc_gpu_performance.csv"
ENABLE_EXACT_METRICS = False
EXACT_KNN_BATCH_SIZE = 1024

# Data paths (match vector-analysis.ipynb/data-analysis.py)
DATA_DIR = os.environ.get("CLIC_DATA_DIR", "/workspace/data")
ARRAY_RECORD_GLOB = os.environ.get(
    "CLIC_ARRAY_RECORD_GLOB", "clic_edm_qq_pf-test.array_record-*"
)
FEATURE_KEY = os.environ.get("CLIC_FEATURE_KEY", "X")
FEATURE_WIDTH = int(os.environ.get("CLIC_FEATURE_WIDTH", "17"))

_FULL_DATA_CACHE: Optional[np.ndarray] = None


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


def time_fgc_gpu(
    data_gpu: torch.Tensor,
    k: int,
    return_results: bool = False,
) -> Tuple[Optional[float], str, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if data_gpu.shape[0] == 0:
        return 0.0, "empty", None, None
    try:
        start = time.perf_counter()
        coordinates = data_gpu.contiguous()
        row_splits = torch.tensor([0, len(data_gpu)], dtype=torch.int64, device="cuda")
        if return_results:
            indices, distances = binned_select_knn(
                k, coordinates, row_splits, direction=None, n_bins=None
            )
        else:
            indices, distances = None, None
            binned_select_knn(k, coordinates, row_splits, direction=None, n_bins=None)
        torch.cuda.synchronize()
        return ms_since(start), "ok", indices, distances
    except Exception as exc:
        print(f"FGC error: {exc}")
        return None, "error", None, None


def exact_knn_cpu(
    data: np.ndarray, k: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    if data.shape[0] == 0:
        return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)
    k = min(k, data.shape[0])
    data_t = torch.from_numpy(data.astype(np.float32, copy=False))
    n_points = data_t.shape[0]
    all_indices = np.empty((n_points, k), dtype=np.int64)
    all_distances = np.empty((n_points, k), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            queries = data_t[start:end]
            distances = torch.cdist(queries, data_t, p=2)
            top_dists, top_indices = torch.topk(
                distances, k, dim=1, largest=False, sorted=True
            )
            all_indices[start:end] = top_indices.cpu().numpy()
            all_distances[start:end] = top_dists.cpu().numpy()
    return all_indices, all_distances


def compute_knn_metrics(
    approx_indices: np.ndarray,
    approx_distances: np.ndarray,
    exact_indices: np.ndarray,
    exact_distances: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    n_queries = exact_indices.shape[0]
    if n_queries == 0:
        return None, None, None
    k = exact_indices.shape[1]
    total_recall = 0.0
    exact_match_count = 0
    max_distance_error: Optional[float] = None
    for i in range(n_queries):
        approx_set = set(int(x) for x in approx_indices[i])
        exact_set = set(int(x) for x in exact_indices[i])
        intersection = approx_set & exact_set
        total_recall += len(intersection) / k if k else 0.0
        if approx_set == exact_set:
            exact_match_count += 1
        if intersection:
            exact_dist_map = {
                int(idx): float(exact_distances[i, j])
                for j, idx in enumerate(exact_indices[i])
            }
            for j, idx in enumerate(approx_indices[i]):
                idx_int = int(idx)
                if idx_int not in exact_dist_map:
                    continue
                err = abs(float(approx_distances[i, j]) - exact_dist_map[idx_int])
                if max_distance_error is None or err > max_distance_error:
                    max_distance_error = err
    recall_at_k = total_recall / n_queries
    exact_match_rate = exact_match_count / n_queries
    return recall_at_k, exact_match_rate, max_distance_error


def ensure_csv_header(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            return
    if existing_header == header:
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})


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
        raise RuntimeError("No runs left in gpu-runs-fgc_gpu.txt")
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
    recall_at_k: Optional[float] = None,
    exact_match_rate: Optional[float] = None,
    max_distance_error: Optional[float] = None,
    include_metrics: bool = False,
) -> None:
    header = ["dim", "points", "k", "time_ms", "status"]
    if include_metrics:
        header += ["recall_at_k", "exact_match_rate", "max_distance_error"]
        ensure_csv_header(path, header)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        row = [dim, points, k, time_ms, status]
        if include_metrics:
            row += [recall_at_k, exact_match_rate, max_distance_error]
        writer.writerow(row)


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

        print(f"Running FGC GPU benchmark: dim={dim}, points={points}, k={k}")
        data_np = load_data(points, dim)
        data_gpu = torch.tensor(data_np, dtype=torch.float32, device="cuda")
        time_ms, status, approx_indices, approx_distances = time_fgc_gpu(
            data_gpu, k, return_results=ENABLE_EXACT_METRICS
        )
        recall_at_k = None
        exact_match_rate = None
        max_distance_error = None
        if (
            ENABLE_EXACT_METRICS
            and status == "ok"
            and approx_indices is not None
            and approx_distances is not None
        ):
            exact_indices, exact_distances = exact_knn_cpu(
                data_np, k, EXACT_KNN_BATCH_SIZE
            )
            recall_at_k, exact_match_rate, max_distance_error = compute_knn_metrics(
                approx_indices.detach().cpu().numpy(),
                approx_distances.detach().cpu().numpy(),
                exact_indices,
                exact_distances,
            )
        append_result(
            output_path,
            dim,
            points,
            k,
            time_ms,
            status,
            recall_at_k=recall_at_k,
            exact_match_rate=exact_match_rate,
            max_distance_error=max_distance_error,
            include_metrics=ENABLE_EXACT_METRICS,
        )
        delete_run_line(runs_path, line_index)
        print(f"Wrote results to {output_path} and removed run line.")
        del data_gpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
