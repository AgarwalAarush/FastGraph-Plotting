#!/usr/bin/env python3
"""
Check numerical sanity of ArrayRecord data used by cuVS CAGRA.

This script mirrors the data loading/slicing used in generate_gpu_cuvs_cagra.py:
- Load ArrayRecord shards using CLIC_* env vars.
- Slice first `dim` columns and first `points` rows.

It prints a concise PASS/WARN/FAIL report per (dim, points) configuration and can
optionally write JSON/CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEED = 42

DATA_DIR = os.path.expanduser(os.environ.get("CLIC_DATA_DIR", "/workspace/data"))
ARRAY_RECORD_GLOB = os.environ.get(
    "CLIC_ARRAY_RECORD_GLOB", "clic_edm_qq_pf-test.array_record-*"
)
FEATURE_KEY = os.environ.get("CLIC_FEATURE_KEY", "X")
FEATURE_WIDTH = int(os.environ.get("CLIC_FEATURE_WIDTH", "17"))

DEFAULT_RUNS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-runs-cuvs_cagra.txt")

_FULL_DATA_CACHE: Optional[np.ndarray] = None


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: Dict[str, object]


def _empty_data(dimensions: int) -> np.ndarray:
    return np.empty((0, dimensions), dtype=np.float32)


def _load_all_from_array_record(data_dir: str, feature_width: int) -> np.ndarray:
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
            return np.empty((0, feature_width), dtype=np.float32)
        x = np.asarray(values, dtype=np.float32)
        if x.size % FEATURE_WIDTH != 0:
            raise ValueError(
                f"Feature '{FEATURE_KEY}' size {x.size} is not divisible by {FEATURE_WIDTH}"
            )
        x = x.reshape(-1, FEATURE_WIDTH)[:, :feature_width]
        return x

    print("Counting total particles...")
    total_particles = 0
    for shard_path in shard_paths:
        with ArrayRecordDataSource(str(shard_path)) as ds:
            for i in range(len(ds)):
                x = decode_example(ds[i])
                total_particles += len(x)

    print(f"Loading {total_particles:,} particles...")
    all_particles = np.zeros((total_particles, feature_width), dtype=np.float32)
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
    if dimensions > FEATURE_WIDTH:
        raise ValueError(f"Requested dim={dimensions} exceeds FEATURE_WIDTH={FEATURE_WIDTH}")
    if _FULL_DATA_CACHE is None:
        print(f"Loading ArrayRecord data from {DATA_DIR}")
        _FULL_DATA_CACHE = _load_all_from_array_record(DATA_DIR, FEATURE_WIDTH)
    full = _FULL_DATA_CACHE
    if n_points >= len(full):
        if n_points > len(full):
            print(f"Requested {n_points} points, using all {len(full)} available.")
        return full[:, :dimensions]
    return full[:n_points, :dimensions]


def parse_int_list(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                start, end = end, start
            items.extend(list(range(start, end + 1)))
        else:
            items.append(int(part))
    return items


def parse_runs_file(path: str) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid run line: {line}")
            runs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return runs


def format_status(result: CheckResult) -> str:
    return f"[{result.status}] {result.name}: {result.message}"


def find_nonfinite_rows(
    x: np.ndarray, max_examples: int, chunk_size: int
) -> List[int]:
    indices: List[int] = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start:start + chunk_size]
        mask = ~np.isfinite(chunk).all(axis=1)
        if mask.any():
            for idx in np.where(mask)[0]:
                indices.append(start + int(idx))
                if len(indices) >= max_examples:
                    return indices
    return indices


def compute_column_stats(
    x: np.ndarray,
    max_abs_threshold: float,
    near_constant_threshold: float,
    dynamic_range_threshold: float,
) -> Tuple[Dict[str, object], List[int], List[int], List[int]]:
    n_cols = x.shape[1]
    stats: Dict[str, List[float]] = {
        "min": [],
        "max": [],
        "mean": [],
        "std": [],
        "min_nonzero_abs": [],
    }
    extreme_cols: List[int] = []
    constant_cols: List[int] = []
    dynamic_range_cols: List[int] = []

    for j in range(n_cols):
        col = x[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            stats["min"].append(float("nan"))
            stats["max"].append(float("nan"))
            stats["mean"].append(float("nan"))
            stats["std"].append(float("nan"))
            stats["min_nonzero_abs"].append(float("nan"))
            continue
        col_finite = col[finite]
        col_min = float(np.min(col_finite))
        col_max = float(np.max(col_finite))
        col_mean = float(np.mean(col_finite))
        col_std = float(np.std(col_finite))
        nonzero = col_finite[np.nonzero(col_finite)]
        min_nonzero_abs = float(np.min(np.abs(nonzero))) if nonzero.size else float("inf")

        stats["min"].append(col_min)
        stats["max"].append(col_max)
        stats["mean"].append(col_mean)
        stats["std"].append(col_std)
        stats["min_nonzero_abs"].append(min_nonzero_abs)

        if max(abs(col_min), abs(col_max)) > max_abs_threshold:
            extreme_cols.append(j)
        if col_std < near_constant_threshold:
            constant_cols.append(j)
        if min_nonzero_abs > 0 and max(abs(col_min), abs(col_max)) / max(1e-12, min_nonzero_abs) > dynamic_range_threshold:
            dynamic_range_cols.append(j)

    return stats, extreme_cols, constant_cols, dynamic_range_cols


def compute_nonfinite_counts(x: np.ndarray) -> Dict[str, object]:
    per_col = []
    total_nan = total_posinf = total_neginf = 0
    for j in range(x.shape[1]):
        col = x[:, j]
        nan = int(np.isnan(col).sum())
        posinf = int(np.isposinf(col).sum())
        neginf = int(np.isneginf(col).sum())
        per_col.append((j, nan, posinf, neginf))
        total_nan += nan
        total_posinf += posinf
        total_neginf += neginf
    per_col_sorted = sorted(
        per_col, key=lambda item: item[1] + item[2] + item[3], reverse=True
    )
    return {
        "total_nan": total_nan,
        "total_posinf": total_posinf,
        "total_neginf": total_neginf,
        "per_col": per_col_sorted,
    }


def sample_rows(x: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= sample_size:
        return x.copy()
    indices = rng.choice(x.shape[0], size=sample_size, replace=False)
    return x[indices]


def compute_overflow_risk(
    x: np.ndarray,
    value_thresholds: Sequence[float],
    l2_thresholds: Sequence[float],
    l2_sample_size: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    counts_above = {thr: 0 for thr in value_thresholds}
    rows_above = {thr: 0 for thr in value_thresholds}
    l2_counts = {thr: 0 for thr in l2_thresholds}
    max_abs = 0.0
    max_l2 = 0.0

    sample = sample_rows(x, l2_sample_size, rng)
    sample_finite = np.where(np.isfinite(sample), sample, 0.0).astype(np.float64, copy=False)
    sample_l2 = np.sum(sample_finite * sample_finite, axis=1)

    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start:start + chunk_size]
        finite = np.isfinite(chunk)
        abs_chunk = np.abs(chunk)
        max_abs = max(max_abs, float(np.nanmax(abs_chunk)))
        for thr in value_thresholds:
            counts_above[thr] += int(np.sum(abs_chunk > thr))
            rows_above[thr] += int(np.sum(np.any(abs_chunk > thr, axis=1)))
        chunk_safe = np.where(finite, chunk, 0.0).astype(np.float64)
        l2sq = np.sum(chunk_safe * chunk_safe, axis=1)
        max_l2 = max(max_l2, float(np.nanmax(l2sq)))
        for thr in l2_thresholds:
            l2_counts[thr] += int(np.sum(l2sq > thr))

    percentiles = {
        "p99": float(np.percentile(sample_l2, 99)),
        "p99_9": float(np.percentile(sample_l2, 99.9)),
    }

    return {
        "max_abs": max_abs,
        "counts_above": counts_above,
        "rows_above": rows_above,
        "max_l2": max_l2,
        "l2_counts": l2_counts,
        "l2_percentiles": percentiles,
        "l2_sample_size": int(sample.shape[0]),
    }


def compute_duplicate_stats(
    x: np.ndarray,
    sample_size: int,
    dup_round: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    sample = sample_rows(x, sample_size, rng)
    finite_mask = np.isfinite(sample).all(axis=1)
    sample = sample[finite_mask]
    if sample.size == 0:
        return {"sample_size": 0, "duplicates": 0, "duplicate_ratio": 0.0}
    rounded = np.round(sample / dup_round) * dup_round
    unique = np.unique(rounded, axis=0)
    duplicates = sample.shape[0] - unique.shape[0]
    ratio = duplicates / max(1, sample.shape[0])
    return {
        "sample_size": int(sample.shape[0]),
        "duplicates": int(duplicates),
        "duplicate_ratio": float(ratio),
    }


def compute_norm_stats(
    x: np.ndarray,
    zero_threshold: float,
    sample_size: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    zero_count = 0
    max_norm = 0.0
    min_norm = float("inf")
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start:start + chunk_size]
        chunk_safe = np.where(np.isfinite(chunk), chunk, 0.0).astype(np.float64)
        norms = np.linalg.norm(chunk_safe, axis=1)
        zero_count += int(np.sum(norms < zero_threshold))
        max_norm = max(max_norm, float(np.max(norms)))
        min_norm = min(min_norm, float(np.min(norms)))

    sample = sample_rows(x, sample_size, rng)
    sample_safe = np.where(np.isfinite(sample), sample, 0.0).astype(np.float64)
    sample_norms = np.linalg.norm(sample_safe, axis=1)
    return {
        "zero_count": int(zero_count),
        "zero_threshold": zero_threshold,
        "min_norm": float(min_norm),
        "max_norm": float(max_norm),
        "median_norm": float(np.median(sample_norms)),
        "p95_norm": float(np.percentile(sample_norms, 95)),
        "p99_norm": float(np.percentile(sample_norms, 99)),
        "sample_size": int(sample.shape[0]),
    }


def compute_column_semantics(
    x: np.ndarray, sample_size: int, monotonic_sample_size: int, rng: np.random.Generator
) -> Dict[int, Dict[str, object]]:
    results: Dict[int, Dict[str, object]] = {}
    sample = sample_rows(x, sample_size, rng)
    monotonic_rows = x[: min(x.shape[0], monotonic_sample_size)]

    for j in range(x.shape[1]):
        col = sample[:, j]
        finite = np.isfinite(col)
        col = col[finite]
        if col.size == 0:
            results[j] = {"integer_like": 0.0, "unique_count": 0, "monotonic_frac": 0.0}
            continue
        integer_like = float(np.mean(np.isclose(col, np.round(col))))
        unique_count = int(np.unique(col).shape[0])

        mono_col = monotonic_rows[:, j]
        mono_col = mono_col[np.isfinite(mono_col)]
        if mono_col.size < 2:
            monotonic_frac = 0.0
        else:
            diffs = np.diff(mono_col)
            monotonic_frac = float(np.mean(diffs >= 0) if diffs.size else 0.0)

        results[j] = {
            "integer_like": integer_like,
            "unique_count": unique_count,
            "monotonic_frac": monotonic_frac,
            "sample_size": int(col.size),
        }
    return results


def quick_knn_sanity(
    x: np.ndarray, k: int, sample_size: int, rng: np.random.Generator
) -> Dict[str, object]:
    if x.shape[0] == 0:
        return {"sample_size": 0, "status": "skip"}
    sample = sample_rows(x, sample_size, rng)
    sample_safe64 = np.where(np.isfinite(sample), sample, 0.0).astype(np.float64)
    sample_safe32 = np.where(np.isfinite(sample), sample, 0.0).astype(np.float32)

    n = sample_safe64.shape[0]
    if n <= k:
        return {"sample_size": n, "status": "skip"}

    invalid_neighbors = 0
    duplicate_neighbors = 0
    float32_nonfinite = 0

    block = 200
    for start in range(0, n, block):
        end = min(n, start + block)
        queries64 = sample_safe64[start:end]
        queries32 = sample_safe32[start:end]

        dist64 = np.sum((queries64[:, None, :] - sample_safe64[None, :, :]) ** 2, axis=2)
        dist32 = np.sum((queries32[:, None, :] - sample_safe32[None, :, :]) ** 2, axis=2)

        if np.isnan(dist32).any() or np.isinf(dist32).any():
            float32_nonfinite += 1

        for i in range(end - start):
            dist64[i, start + i] = np.inf
            dist32[i, start + i] = np.inf
        idx = np.argpartition(dist64, kth=k, axis=1)[:, :k]
        for row in idx:
            if np.any(row < 0) or np.any(row >= n):
                invalid_neighbors += 1
            if np.unique(row).shape[0] < row.shape[0]:
                duplicate_neighbors += 1

    return {
        "sample_size": n,
        "invalid_neighbors": int(invalid_neighbors),
        "duplicate_neighbors": int(duplicate_neighbors),
        "float32_nonfinite_blocks": int(float32_nonfinite),
        "status": "ok",
    }


def recommend_transform(x: np.ndarray, clip: float) -> Dict[str, object]:
    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    mean = np.mean(safe, axis=0)
    std = np.std(safe, axis=0)
    return {
        "clip": clip,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "snippet": (
            "X = X.astype(np.float32)\n"
            "X = np.nan_to_num(X, nan=0.0, posinf=clip, neginf=-clip)\n"
            "X = np.clip(X, -clip, clip)\n"
            "# Optional standardization:\n"
            "# X = (X - mean) / np.maximum(std, 1e-12)\n"
        ),
    }


def apply_fix(x: np.ndarray, clip: float) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(x, -clip, clip)


def analyze_config(
    dim: int,
    points: int,
    k: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[CheckResult], Dict[str, object]]:
    data = load_data(points, dim)
    if args.fix:
        data = apply_fix(data, args.clip)
    summary: Dict[str, object] = {"dim": dim, "points": points, "k": k}
    results: List[CheckResult] = []

    # Basic info
    order = "C" if data.flags["C_CONTIGUOUS"] else ("F" if data.flags["F_CONTIGUOUS"] else "non-contiguous")
    dtype_ok = data.dtype == np.float32
    results.append(
        CheckResult(
            name="basic",
            status="PASS" if dtype_ok else "WARN",
            message=f"shape={data.shape}, dtype={data.dtype}, order={order}, device=numpy",
            details={
                "shape": data.shape,
                "dtype": str(data.dtype),
                "order": order,
                "device": "numpy",
            },
        )
    )

    # Non-finite
    nonfinite = compute_nonfinite_counts(data)
    total_nonfinite = nonfinite["total_nan"] + nonfinite["total_posinf"] + nonfinite["total_neginf"]
    nonfinite_rows = find_nonfinite_rows(data, args.max_nonfinite_examples, args.chunk_size)
    status = "PASS" if total_nonfinite == 0 else "FAIL"
    results.append(
        CheckResult(
            name="non_finite",
            status=status,
            message=f"NaN={nonfinite['total_nan']}, +Inf={nonfinite['total_posinf']}, -Inf={nonfinite['total_neginf']}",
            details={
                **nonfinite,
                "example_rows": nonfinite_rows,
                "top_columns": nonfinite["per_col"][: args.top_k_columns],
            },
        )
    )

    # Column stats
    stats, extreme_cols, constant_cols, dynamic_cols = compute_column_stats(
        data,
        max_abs_threshold=args.max_abs_threshold,
        near_constant_threshold=args.near_constant_threshold,
        dynamic_range_threshold=args.dynamic_range_threshold,
    )
    status = "PASS"
    if extreme_cols or dynamic_cols:
        status = "WARN"
    if len(extreme_cols) > 0:
        status = "FAIL"
    results.append(
        CheckResult(
            name="column_stats",
            status=status,
            message=(
                f"extreme_cols={len(extreme_cols)}, constant_cols={len(constant_cols)}, "
                f"dynamic_range_cols={len(dynamic_cols)}"
            ),
            details={
                "extreme_cols": extreme_cols[: args.top_k_columns],
                "constant_cols": constant_cols[: args.top_k_columns],
                "dynamic_range_cols": dynamic_cols[: args.top_k_columns],
                "stats": stats,
            },
        )
    )

    # Overflow risk
    overflow = compute_overflow_risk(
        data,
        value_thresholds=args.value_thresholds,
        l2_thresholds=args.l2_thresholds,
        l2_sample_size=args.l2_sample_size,
        chunk_size=args.chunk_size,
        rng=rng,
    )
    overflow_status = "PASS"
    if overflow["max_abs"] > args.value_thresholds[-1]:
        overflow_status = "FAIL"
    elif overflow["max_abs"] > args.value_thresholds[0]:
        overflow_status = "WARN"
    results.append(
        CheckResult(
            name="overflow_risk",
            status=overflow_status,
            message=f"max_abs={overflow['max_abs']:.3e}, max_l2={overflow['max_l2']:.3e}, p99.9_l2={overflow['l2_percentiles']['p99_9']:.3e}",
            details=overflow,
        )
    )

    # Duplicate / norms
    duplicates = compute_duplicate_stats(
        data,
        sample_size=args.dup_sample_size,
        dup_round=args.dup_round,
        rng=rng,
    )
    dup_status = "PASS" if duplicates["duplicate_ratio"] < args.dup_warn_ratio else "WARN"
    results.append(
        CheckResult(
            name="duplicates",
            status=dup_status,
            message=f"sample_duplicates={duplicates['duplicates']} ({duplicates['duplicate_ratio']:.4f})",
            details=duplicates,
        )
    )

    norms = compute_norm_stats(
        data,
        zero_threshold=args.zero_threshold,
        sample_size=args.norm_sample_size,
        chunk_size=args.chunk_size,
        rng=rng,
    )
    norm_status = "PASS" if norms["zero_count"] == 0 else "WARN"
    results.append(
        CheckResult(
            name="norms",
            status=norm_status,
            message=f"zero_norm_count={norms['zero_count']}, min_norm={norms['min_norm']:.3e}, max_norm={norms['max_norm']:.3e}",
            details=norms,
        )
    )

    # Column semantics
    semantics = compute_column_semantics(
        data,
        sample_size=args.semantic_sample_size,
        monotonic_sample_size=args.monotonic_sample_size,
        rng=rng,
    )
    flagged_cols = [
        j for j, info in semantics.items()
        if info["integer_like"] > args.integer_like_threshold
        or info["monotonic_frac"] > args.monotonic_threshold
    ]
    sem_status = "WARN" if flagged_cols else "PASS"
    results.append(
        CheckResult(
            name="column_semantics",
            status=sem_status,
            message=f"flagged_cols={len(flagged_cols)}",
            details={"flagged_cols": flagged_cols[: args.top_k_columns], "semantics": semantics},
        )
    )

    # Quick kNN sanity
    if args.knn:
        knn = quick_knn_sanity(
            data,
            k=args.knn_k,
            sample_size=args.knn_sample_size,
            rng=rng,
        )
        knn_status = "PASS"
        if knn.get("invalid_neighbors", 0) or knn.get("duplicate_neighbors", 0):
            knn_status = "WARN"
        if knn.get("float32_nonfinite_blocks", 0) > 0:
            knn_status = "FAIL"
        results.append(
            CheckResult(
                name="knn_sanity",
                status=knn_status,
                message=(
                    f"invalid_neighbors={knn.get('invalid_neighbors', 0)}, "
                    f"duplicate_neighbors={knn.get('duplicate_neighbors', 0)}, "
                    f"float32_nonfinite_blocks={knn.get('float32_nonfinite_blocks', 0)}"
                ),
                details=knn,
            )
        )

    summary["results"] = [r.__dict__ for r in results]
    summary["recommendation"] = recommend_transform(data, args.clip)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for CAGRA input data.")
    parser.add_argument("--runs-file", default=DEFAULT_RUNS_FILE, help="Runs file (dim,points,k).")
    parser.add_argument("--dims", type=str, default="", help="Comma-separated dims (e.g. 3,5,10 or 3-10).")
    parser.add_argument("--points", type=str, default="", help="Comma-separated point counts.")
    parser.add_argument("--k", type=int, default=40, help="k for quick kNN when runs file not used.")
    parser.add_argument("--write-json", type=str, default="", help="Write JSON summary to path.")
    parser.add_argument("--write-csv", type=str, default="", help="Write CSV summary to path.")
    parser.add_argument("--fix", action="store_true", help="Apply safe cleaning to data before checks.")

    parser.add_argument("--sample-size", type=int, default=200000, help="Default sample size.")
    parser.add_argument("--dup-sample-size", type=int, default=200000, help="Sample size for duplicates.")
    parser.add_argument("--dup-round", type=float, default=1e-6, help="Rounding step for duplicates.")
    parser.add_argument("--norm-sample-size", type=int, default=200000, help="Sample size for norm stats.")
    parser.add_argument("--semantic-sample-size", type=int, default=200000, help="Sample size for semantics.")
    parser.add_argument("--monotonic-sample-size", type=int, default=100000, help="Rows for monotonic check.")
    parser.add_argument("--l2-sample-size", type=int, default=200000, help="Sample size for L2 percentiles.")
    parser.add_argument("--knn", action="store_true", help="Run quick kNN sanity check.")
    parser.add_argument("--knn-sample-size", type=int, default=2000, help="Sample size for quick kNN.")
    parser.add_argument("--knn-k", type=int, default=40, help="k for quick kNN check.")

    parser.add_argument("--max-abs-threshold", type=float, default=1e6, help="Extreme abs value threshold.")
    parser.add_argument("--near-constant-threshold", type=float, default=1e-12, help="Std threshold for near-constant.")
    parser.add_argument("--dynamic-range-threshold", type=float, default=1e8, help="Dynamic range threshold.")
    parser.add_argument("--integer-like-threshold", type=float, default=0.99, help="Integer-like column threshold.")
    parser.add_argument("--monotonic-threshold", type=float, default=0.99, help="Monotonic column threshold.")
    parser.add_argument("--zero-threshold", type=float, default=1e-12, help="Zero/near-zero norm threshold.")
    parser.add_argument("--clip", type=float, default=1e6, help="Clip value for --fix.")
    parser.add_argument("--max-nonfinite-examples", type=int, default=10, help="Max non-finite row examples.")
    parser.add_argument("--top-k-columns", type=int, default=5, help="Top columns to report.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Chunk size for streaming stats.")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    runs: List[Tuple[int, int, int]] = []
    if args.dims and args.points:
        dims = parse_int_list(args.dims)
        points = parse_int_list(args.points)
        for dim in dims:
            for pt in points:
                runs.append((dim, pt, args.k))
    else:
        if not os.path.exists(args.runs_file):
            raise FileNotFoundError(f"Runs file not found: {args.runs_file}")
        runs = parse_runs_file(args.runs_file)

    args.value_thresholds = (1e10, 1e15, 1e19)
    args.l2_thresholds = (1e30, 1e35, 1e38)

    summaries: List[Dict[str, object]] = []
    for dim, points, k in runs:
        print(f"\n=== Config dim={dim}, points={points}, k={k} ===")
        results, summary = analyze_config(dim, points, k, args, rng)
        for result in results:
            print(format_status(result))
        summaries.append(summary)

    if args.write_json:
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Wrote JSON summary to {args.write_json}")

    if args.write_csv:
        fieldnames = ["dim", "points", "k", "check", "status", "message"]
        with open(args.write_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                for result in summary["results"]:
                    writer.writerow(
                        {
                            "dim": summary["dim"],
                            "points": summary["points"],
                            "k": summary["k"],
                            "check": result["name"],
                            "status": result["status"],
                            "message": result["message"],
                        }
                    )
        print(f"Wrote CSV summary to {args.write_csv}")


if __name__ == "__main__":
    main()
