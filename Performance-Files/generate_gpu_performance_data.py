#!/usr/bin/env python3
"""
GPU benchmarking runner for FAISS (GPU), GGNN, and FGC (GPU).
Consumes the first line of gpu-runs.txt (CSV: dim,points,k) and appends results
to gpu_performance.csv (one line per method), then removes the processed run.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from fastgraphcompute import binned_select_knn


SEED = 42
RUNS_FILE = "gpu-runs.txt"
OUTPUT_CSV = "gpu_performance.csv"

# GGNN parameters (mirroring generate_ggnn_performance.py defaults)
K_BUILD = 24
TAU_BUILD = 0.5
TAU_QUERY = 0.64
MAX_ITERATIONS = 400


def generate_data(n_points: int, dimensions: int, seed_offset: int = 0) -> np.ndarray:
    np.random.seed(SEED + seed_offset)
    return np.random.randn(n_points, dimensions).astype(np.float32)


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def time_faiss_gpu(data: np.ndarray, k: int) -> Tuple[Optional[float], str]:
    if data.shape[0] == 0:
        return 0.0, "empty"
    try:
        import faiss
    except ImportError:
        return None, "not_installed"

    try:
        start = time.perf_counter()
        index = faiss.IndexFlatL2(data.shape[1])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(data)
        index.search(data, k)
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"FAISS error: {exc}")
        return None, "error"


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
        my_ggnn.build(k_build=K_BUILD, tau_build=TAU_BUILD, measure=ggnn.DistanceMeasure.Euclidean)
        my_ggnn.query(data_gpu, k, TAU_QUERY, MAX_ITERATIONS, ggnn.DistanceMeasure.Euclidean)
        torch.cuda.synchronize()
        return ms_since(start), "ok"
    except Exception as exc:
        print(f"GGNN error: {exc}")
        return None, "error"


def time_fgc_gpu(data_gpu: torch.Tensor, k: int) -> Tuple[Optional[float], str]:
    if data_gpu.shape[0] == 0:
        return 0.0, "empty"
    try:
        start = time.perf_counter()
        coordinates = data_gpu.contiguous()
        row_splits = torch.tensor([0, len(data_gpu)], dtype=torch.int64, device="cuda")
        binned_select_knn(k, coordinates, row_splits, direction=None, n_bins=None)
        torch.cuda.synchronize()
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
        raise RuntimeError("No runs left in gpu-runs.txt")
    return parse_run_line(lines[idx]), idx


def delete_run_line(path: str, line_index: int) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return
    remaining = lines[:line_index] + lines[line_index + 1 :]
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
            writer.writerow(["dim", "points", "k", "method", "time_ms", "status"])
        for method, (time_ms, status) in results.items():
            writer.writerow([dim, points, k, method, time_ms, status])


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(base_dir, RUNS_FILE)
    output_path = os.path.join(base_dir, OUTPUT_CSV)

    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"{RUNS_FILE} not found in {base_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU benchmarking")

    try:
        (dim, points, k), line_index = read_next_run(runs_path)
    except RuntimeError as exc:
        print(str(exc))
        return

    print(f"Running GPU benchmarks: dim={dim}, points={points}, k={k}")
    data_np = generate_data(points, dim)
    data_gpu = torch.tensor(data_np, dtype=torch.float32, device="cuda")

    results: Dict[str, Tuple[Optional[float], str]] = {
        "faiss_gpu": time_faiss_gpu(data_np, k),
        "ggnn": time_ggnn(data_gpu, k),
        "fgc_gpu": time_fgc_gpu(data_gpu, k),
    }

    append_results(output_path, dim, points, k, results)
    delete_run_line(runs_path, line_index)
    print(f"Wrote results to {output_path} and removed run line.")


if __name__ == "__main__":
    main()
