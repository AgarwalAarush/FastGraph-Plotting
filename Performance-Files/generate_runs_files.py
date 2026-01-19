#!/usr/bin/env python3
"""
Generate run combinations for CPU/GPU benchmarks.
Outputs all-runs.txt, cpu-runs.txt, and gpu-runs.txt in this directory.
Each line is CSV: dim,points,k (no header).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple


RUNS_FILENAME = "all-runs.txt"
CPU_RUNS_FILENAME = "cpu-runs.txt"
GPU_RUNS_FILENAME = "gpu-runs.txt"


def points_list() -> List[int]:
    return [
        0,
        100_000,
        500_000,
        1_000_000,
        1_500_000,
        2_000_000,
        2_500_000,
        3_000_000,
        3_500_000,
        4_000_000,
        4_500_000,
        5_000_000,
    ]


def add_combos(
    combos: List[Tuple[int, int, int]],
    seen: set[Tuple[int, int, int]],
    new_combos: Iterable[Tuple[int, int, int]],
) -> None:
    for combo in new_combos:
        if combo in seen:
            continue
        seen.add(combo)
        combos.append(combo)


def build_combinations() -> List[Tuple[int, int, int]]:
    combos: List[Tuple[int, int, int]] = []
    seen: set[Tuple[int, int, int]] = set()

    points = points_list()

    # 1) dim=3, k=40, varying points
    add_combos(combos, seen, ((3, p, 40) for p in points))

    # 2) dim=5, k=40, varying points
    add_combos(combos, seen, ((5, p, 40) for p in points))

    # 3) dim=3, k in {10,40,100}, varying points
    for k in (10, 40, 100):
        add_combos(combos, seen, ((3, p, k) for p in points))

    # 4) dim in [2..15], points=5M, k=40
    add_combos(combos, seen, ((d, 5_000_000, 40) for d in range(2, 16)))

    return combos


def write_runs_file(path: str, combos: List[Tuple[int, int, int]], repeats: int = 10) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(repeats):
            for dim, points, k in combos:
                f.write(f"{dim},{points},{k}\n")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_runs_path = os.path.join(base_dir, RUNS_FILENAME)
    cpu_runs_path = os.path.join(base_dir, CPU_RUNS_FILENAME)
    gpu_runs_path = os.path.join(base_dir, GPU_RUNS_FILENAME)

    combos = build_combinations()
    write_runs_file(all_runs_path, combos, repeats=10)

    # Duplicate to CPU/GPU run queues.
    for target_path in (cpu_runs_path, gpu_runs_path):
        with open(all_runs_path, "r", encoding="utf-8") as src:
            content = src.read()
        with open(target_path, "w", encoding="utf-8") as dst:
            dst.write(content)

    print(f"Wrote {all_runs_path}")
    print(f"Duplicated to {cpu_runs_path} and {gpu_runs_path}")


if __name__ == "__main__":
    main()
