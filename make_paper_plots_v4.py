#!/usr/bin/env python3
"""
make_paper_plots_v4.py — generate the 13 figures expected by
Performance-Paper-PCA/Paper.tex v4.

Reads canonical CSVs from ~/Performance/, writes PNGs to a configurable
output directory (default: ~/FastGraph-Plotting/v4-plots/).

Must run via srun on falcon (loads CSVs into pandas; head-node watchdog
kills > 4 GB).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────

PERF = Path(os.environ.get("PERF_DIR", os.path.expanduser("~/Performance")))
OUT = Path(os.environ.get("OUT_DIR", os.path.expanduser("~/FastGraph-Plotting/v4-plots")))
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Backend palette
# ──────────────────────────────────────────────────────────────────────────

# Keep PCA-FGC visually dominant (red, thicker). Vanilla FGC related (orange).
# Exact-GPU baselines in cool colors; approximate methods in muted grays/browns.
BACKEND = {
    "pca_fgc":   dict(label="FastGraph (PCA)", color="#D62728", marker="o", lw=2.4),
    "fgc":       dict(label="FastGraph (axis)", color="#FF7F0E", marker="s", lw=1.6),
    "faiss":     dict(label="FAISS-GPU (exact)", color="#2CA02C", marker="D", lw=1.6),
    "cuvs_bf":   dict(label="cuVS BF (exact)",   color="#1F77B4", marker="^", lw=1.6),
    "cagra_nnd": dict(label="CAGRA-nnd (approx)", color="#8C564B", marker="v", lw=1.4),
    "ggnn":      dict(label="GGNN (approx)",     color="#7F7F7F", marker="x", lw=1.4),
    # CLOVER variants
    "bitonic":   dict(label="CLOVER bitonic", color="#9467BD", marker="P", lw=1.6),
    "warpwise":  dict(label="CLOVER warpwise", color="#17BECF", marker="*", lw=1.6),
    "hubs":      dict(label="CLOVER hubs", color="#1A55A3", marker="X", lw=1.8),
}

# ──────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────

def load_ok(path: Path, status_col: str = "status") -> pd.DataFrame:
    df = pd.read_csv(path)
    if status_col in df.columns:
        df = df[df[status_col] == "ok"].copy()
    return df


def agg_median(df: pd.DataFrame, group_cols: Sequence[str], val: str = "time_ms") -> pd.DataFrame:
    """Median + interquartile range as error bars.

    Using IQR (Q1, Q3) rather than min/max keeps single-rep outliers from
    visually dominating the error bars. Median is the central estimator
    used everywhere else in the paper, so the bar captures the bulk of
    the per-cell distribution honestly.
    """
    grp = df.groupby(list(group_cols))[val]
    out = grp.agg(
        median="median",
        lo=lambda s: s.quantile(0.25),
        hi=lambda s: s.quantile(0.75),
        n="count",
    ).reset_index()
    out["err_lo"] = (out["median"] - out["lo"]).clip(lower=0)
    out["err_hi"] = (out["hi"] - out["median"]).clip(lower=0)
    return out


# Production HGCAL CSVs (v5 = merge of v4 baseline + extended mps:100 runs,
# with drift-aware policy: cuVS BF / PCA-FGC fully merged, vanilla FGC /
# FAISS / GGNN use extended-only on cells where the v4 baseline disagreed
# with the clean extended median by >10%. See merge_baseline_extended.py
# in Performance/ for the policy.)
def load_hgcal_timing() -> Dict[str, pd.DataFrame]:
    sources = {
        "pca_fgc":   "gpu_fgc_pca_v5.csv",
        "fgc":       "gpu_fgc_gpu_v5.csv",
        "faiss":     "gpu_faiss_gpu_v5.csv",
        "cuvs_bf":   "gpu_cuvs_bf_v5.csv",
        "ggnn":      "gpu_ggnn_v5.csv",
        "cagra_nnd": "cagra_nn_descent_v5.csv",
    }
    out = {}
    for key, fn in sources.items():
        p = PERF / fn
        if not p.exists():
            print(f"  [warn] {fn} missing; skipping {key}")
            continue
        df = load_ok(p)
        df = df.astype({"dim": int, "points": int, "k": int})
        out[key] = df
    return out


# Synthetic Gaussian CSVs (Binary A)
def load_synth_timing() -> Dict[str, pd.DataFrame]:
    out = {}
    p = PERF / "synthetic_pca_benchmark_binA.csv"
    if p.exists():
        df = load_ok(p).astype({"dim": int, "points": int, "k": int})
        out["pca_fgc"] = df[df["variant"] == "fgc_pca"].copy()
        out["fgc"] = df[df["variant"] == "fgc_vanilla"].copy()
    p = PERF / "synth_gpu_baselines.csv"
    if p.exists():
        df = load_ok(p).astype({"dim": int, "points": int, "k": int})
        for be_name, df_be in df.groupby("backend"):
            # backend column uses "faiss_gpu", "cuvs_bf", "cagra_nnd"
            key = {"faiss_gpu": "faiss", "cuvs_bf": "cuvs_bf", "cagra_nnd": "cagra_nnd"}.get(be_name)
            if key:
                out[key] = df_be.copy()
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────

def _plot_lines_by_backend(ax, dfs: Dict[str, pd.DataFrame], x_col: str, y_col: str,
                            backends_in_order: Iterable[str], log_y: bool = True,
                            errorbars: bool = True) -> None:
    for be in backends_in_order:
        df = dfs.get(be)
        if df is None or df.empty:
            continue
        df = df.sort_values(x_col)
        style = BACKEND[be]
        if errorbars and "err_lo" in df.columns:
            ax.errorbar(df[x_col], df["median"],
                        yerr=[df["err_lo"], df["err_hi"]],
                        label=style["label"], color=style["color"],
                        marker=style["marker"], lw=style["lw"], capsize=2)
        else:
            yvals = df[y_col] if y_col in df.columns else df["median"]
            ax.plot(df[x_col], yvals, label=style["label"], color=style["color"],
                    marker=style["marker"], lw=style["lw"])
    if log_y:
        ax.set_yscale("log")


def _format_n_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x < 1_000_000 else f"{int(x/1_000_000)}M"))


# ──────────────────────────────────────────────────────────────────────────
# Plot 1: gpu_fgc_dimensional_scaling_5M_k40_all_algorithms.png
# ──────────────────────────────────────────────────────────────────────────

def plot_hgcal_dim_scaling_5M(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    N, k = 5_000_000, 40
    fig, ax = plt.subplots(figsize=(8, 5))
    agg_dfs = {}
    for be, df in dfs.items():
        sub = df[(df["points"] == N) & (df["k"] == k)]
        if sub.empty: continue
        agg_dfs[be] = agg_median(sub, ["dim"]).rename(columns={"dim": "x"}).assign(x=lambda d: d["x"])
    for be, a in agg_dfs.items():
        a.rename(columns={"x": "dim"}, inplace=True)
    _plot_lines_by_backend(ax, agg_dfs, "dim", "median",
                            ["pca_fgc", "fgc", "cuvs_bf", "faiss", "cagra_nnd", "ggnn"])
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_title(f"HGCAL kNN graph build — $N{{=}}5\\mathrm{{M}}$, $k{{=}}{k}$")
    ax.legend(loc="best", ncol=2)
    ax.set_xticks(range(2, 11))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 2: gpu_fgc_k_comparison_1M_d2-10_all_algorithms.png
# ──────────────────────────────────────────────────────────────────────────

def plot_hgcal_k_comparison_1M(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    N = 1_000_000
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, k in zip(axes, (10, 40, 100)):
        agg_dfs = {}
        for be, df in dfs.items():
            sub = df[(df["points"] == N) & (df["k"] == k)]
            if sub.empty: continue
            agg_dfs[be] = agg_median(sub, ["dim"])
        _plot_lines_by_backend(ax, agg_dfs, "dim", "median",
                                ["pca_fgc", "fgc", "cuvs_bf", "faiss", "cagra_nnd", "ggnn"])
        ax.set_xlabel("Dimension $d$")
        ax.set_title(f"$k{{=}}{k}$")
        ax.set_xticks(range(2, 11))
    axes[0].set_ylabel("Wall-clock (ms)")
    axes[-1].legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle(f"HGCAL — $N{{=}}1\\mathrm{{M}}$, varying $k$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 3, 4: gpu_fgc_speedup_d{3,8}_all_algorithms.png
# ──────────────────────────────────────────────────────────────────────────

def plot_hgcal_size_scaling(dfs: Dict[str, pd.DataFrame], dim: int, out_path: Path,
                             title_suffix: str = "") -> None:
    k = 40
    fig, ax = plt.subplots(figsize=(8, 5))
    agg_dfs = {}
    for be, df in dfs.items():
        sub = df[(df["dim"] == dim) & (df["k"] == k)]
        if sub.empty: continue
        agg_dfs[be] = agg_median(sub, ["points"]).rename(columns={"points": "points"})
    _plot_lines_by_backend(ax, agg_dfs, "points", "median",
                            ["pca_fgc", "fgc", "cuvs_bf", "faiss", "cagra_nnd", "ggnn"])
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_title(f"HGCAL — $d{{=}}{dim}$, $k{{=}}{k}$ {title_suffix}".strip())
    ax.legend(loc="best", ncol=2)
    _format_n_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 5: synth_dimensional_scaling_1M_k40_gpu.png
# ──────────────────────────────────────────────────────────────────────────

def plot_synth_dim_scaling(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    N, k = 1_000_000, 40
    fig, ax = plt.subplots(figsize=(8, 5))
    agg_dfs = {}
    for be, df in dfs.items():
        sub = df[(df["points"] == N) & (df["k"] == k)]
        if sub.empty: continue
        agg_dfs[be] = agg_median(sub, ["dim"])
    _plot_lines_by_backend(ax, agg_dfs, "dim", "median",
                            ["pca_fgc", "fgc", "cuvs_bf", "faiss", "cagra_nnd"])
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_title(f"Isotropic Gaussian $\\mathcal{{N}}(0,I_d)$ — $N{{=}}1\\mathrm{{M}}$, $k{{=}}{k}$")
    ax.legend(loc="best", ncol=2)
    ax.set_xticks(range(2, 11))
    # Annotate crossover at d=7
    ax.axvline(7, ls=":", color="k", alpha=0.4)
    ax.text(7.1, ax.get_ylim()[1]*0.5, "crossover\n($d{=}7$)", fontsize=9, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 6, 7: synth_speedup_d{3,8}_gpu.png
# ──────────────────────────────────────────────────────────────────────────

def plot_synth_size_scaling(dfs: Dict[str, pd.DataFrame], dim: int, out_path: Path) -> None:
    k = 40
    fig, ax = plt.subplots(figsize=(8, 5))
    agg_dfs = {}
    for be, df in dfs.items():
        sub = df[(df["dim"] == dim) & (df["k"] == k)]
        if sub.empty: continue
        agg_dfs[be] = agg_median(sub, ["points"])
    _plot_lines_by_backend(ax, agg_dfs, "points", "median",
                            ["pca_fgc", "fgc", "cuvs_bf", "faiss", "cagra_nnd"])
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_title(f"Isotropic Gaussian — $d{{=}}{dim}$, $k{{=}}{k}$")
    ax.legend(loc="best", ncol=2)
    _format_n_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 8: clover_headtohead_d3.png — 3 panels (HGCAL / Gaussian / uniform)
# ──────────────────────────────────────────────────────────────────────────

def plot_clover_headtohead(out_path: Path) -> None:
    p_clover = PERF / "clover_headtohead.csv"
    p_pcafgc = PERF / "pca_fgc_clover_headtohead.csv"
    if not p_clover.exists() or not p_pcafgc.exists():
        print(f"  [skip] {out_path.name} — missing CSV inputs")
        return
    cl = pd.read_csv(p_clover)
    pf = load_ok(p_pcafgc)
    # CLOVER time is in ns
    cl["time_ms"] = cl["time_ns"] / 1e6
    # Extract dataset (hgcal / synthgauss / synthuniform) from filename "<N>-<ds>.txt"
    cl["dataset"] = cl["filename"].str.extract(r"-(\w+)\.txt$")[0]
    # Normalize dataset names
    ds_norm = {"hgcal": "hgcal", "synthgauss": "gauss", "synthuniform": "uniform"}
    cl["dataset"] = cl["dataset"].map(ds_norm)
    pf["dataset"] = pf["dataset"].map({"hgcal": "hgcal", "gauss": "gauss",
                                       "synthgauss": "gauss", "synth_gauss": "gauss",
                                       "uniform": "uniform", "synthuniform": "uniform",
                                       "synth_uniform": "uniform"}).fillna(pf["dataset"])

    cl["n"] = cl["n"].astype(int)
    pf["points"] = pf["points"].astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    titles = {"hgcal": "HGCAL", "gauss": "Gaussian $\\mathcal{N}(0,I_3)$", "uniform": "Uniform $U[0,1]^3$"}
    for ax, ds_key in zip(axes, ("hgcal", "gauss", "uniform")):
        # CLOVER lines (per algorithm)
        cl_ds = cl[cl["dataset"] == ds_key]
        for alg in ("bitonic", "warpwise", "hubs"):
            sub = cl_ds[cl_ds["algorithm"] == alg]
            if sub.empty: continue
            agg = sub.groupby("n")["time_ms"].median().reset_index().sort_values("n")
            style = BACKEND[alg]
            ax.plot(agg["n"], agg["time_ms"], label=style["label"],
                    color=style["color"], marker=style["marker"], lw=style["lw"])
        # PCA-FGC + vanilla FGC lines
        pf_ds = pf[pf["dataset"] == ds_key]
        for var, be_key in (("pca", "pca_fgc"), ("vanilla", "fgc")):
            sub = pf_ds[pf_ds["variant"] == var]
            if sub.empty: continue
            agg = sub.groupby("points")["time_ms"].median().reset_index().sort_values("points")
            style = BACKEND[be_key]
            ax.plot(agg["points"], agg["time_ms"], label=style["label"],
                    color=style["color"], marker=style["marker"], lw=style["lw"])
        ax.set_xlabel("Number of points $N$")
        ax.set_title(titles[ds_key])
        ax.set_yscale("log")
        _format_n_axis(ax)
    axes[0].set_ylabel("Wall-clock (ms)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("CLOVER head-to-head at $d{=}3$, $k{=}40$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 9: memory_footprint.png
# ──────────────────────────────────────────────────────────────────────────

def plot_memory(out_path: Path) -> None:
    p = PERF / "memory_usage.csv"
    if not p.exists():
        print(f"  [skip] memory_usage.csv missing"); return
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if "memory_mb" not in df.columns:
        for c in ("mem_mb", "gpu_memory_mb"):
            if c in df.columns: df = df.rename(columns={c: "memory_mb"}); break
    df = df.astype({"dim": int, "points": int, "k": int})
    # Map algorithm names to our palette keys
    alg_map = {
        "fgc": "fgc", "fastgraph": "fgc", "fgc_pca": "pca_fgc", "pca_fgc": "pca_fgc",
        "faiss": "faiss", "faiss_gpu": "faiss",
        "cuvs_bf": "cuvs_bf", "cuvs": "cuvs_bf",
        "ggnn": "ggnn", "cagra": "cagra_nnd", "cagra_nnd": "cagra_nnd",
    }
    df["be"] = df["algorithm"].str.lower().map(alg_map).fillna(df["algorithm"])
    # Sweep over N at d=3, k=40
    d, k = 3, 40
    sub = df[(df["dim"] == d) & (df["k"] == k)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for be in ("pca_fgc", "fgc", "faiss", "cuvs_bf", "cagra_nnd", "ggnn"):
        s = sub[sub["be"] == be]
        if s.empty: continue
        agg = s.groupby("points")["memory_mb"].median().reset_index().sort_values("points")
        style = BACKEND.get(be, dict(label=be, color="k", marker="o", lw=1))
        ax.plot(agg["points"], agg["memory_mb"], label=style["label"],
                color=style["color"], marker=style["marker"], lw=style["lw"])
    ax.set_xlabel("Number of points $N$")
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title(f"Resident GPU memory — $d{{=}}{d}$, $k{{=}}{k}$")
    ax.legend(loc="best", ncol=2)
    _format_n_axis(ax)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Plots 10–13: recall_*.png
# ──────────────────────────────────────────────────────────────────────────

def _load_recall_dist() -> pd.DataFrame:
    """Returns concatenated dataframe with columns dim/points/k/time_ms/recall/be.
    For backends with quality knobs (cuvs itopk_size, ggnn tau_query), keeps
    only the highest-recall row per (dim,points,k) cell.
    """
    parts = []
    for be, fn, qual in (("pca_fgc", "recall_dist_fgc.csv", None),
                          ("faiss", "recall_dist_faiss.csv", None),
                          ("cuvs_bf", "recall_dist_cuvs.csv", "itopk_size"),
                          ("ggnn", "recall_dist_ggnn.csv", "tau_query")):
        p = PERF / fn
        if not p.exists(): continue
        df = pd.read_csv(p)
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()
        # canonical recall column
        if "recall_dist" in df.columns:
            df["recall"] = df["recall_dist"]
        elif "recall_set" in df.columns:
            df["recall"] = df["recall_set"]
        df["be"] = be
        if qual and qual in df.columns:
            # keep highest-recall row per cell
            df = df.sort_values("recall", ascending=False)
            df = df.drop_duplicates(subset=["dim", "points", "k"], keep="first")
        parts.append(df[["dim", "points", "k", "time_ms", "recall", "be"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_recall_exactness(out_path: Path) -> None:
    df = _load_recall_dist()
    if df.empty: print("  [skip] recall_exactness — no data"); return
    # Use d=3 N=500k k=40 representative cell
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.astype({"dim": int, "points": int, "k": int})
    sub = df[(df["dim"] == 3) & (df["points"] == 500_000) & (df["k"] == 40)]
    if sub.empty:
        # fallback: any d=3, k=40
        sub = df[(df["dim"] == 3) & (df["k"] == 40)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Use FGC as the reference (recall_dist=1.0 expected)
    be_order = ["pca_fgc", "faiss", "cuvs_bf", "ggnn"]
    means = []
    labels = []
    colors = []
    for be in be_order:
        s = sub[sub["be"] == be]
        if s.empty: continue
        means.append(s["recall"].mean() if "recall" in s.columns else float("nan"))
        labels.append(BACKEND[be]["label"])
        colors.append(BACKEND[be]["color"])
    bars = ax.bar(range(len(means)), means, color=colors)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Distance-based recall")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", alpha=0.4)
    ax.set_title("Distance-based recall (d=3, N=500k, k=40)")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_recall_controlled(out_path: Path) -> None:
    # Stub: bar chart of best-quality recall per backend
    df = _load_recall_dist()
    if df.empty: print("  [skip] recall_controlled — no data"); return
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.astype({"dim": int, "points": int, "k": int})
    sub = df[(df["points"] == 500_000) & (df["k"] == 40)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for be in ("pca_fgc", "faiss", "cuvs_bf", "ggnn"):
        s = sub[sub["be"] == be]
        if s.empty: continue
        agg = s.groupby("dim")["recall"].mean().reset_index().sort_values("dim")
        style = BACKEND[be]
        ax.plot(agg["dim"], agg["recall"], label=style["label"],
                color=style["color"], marker=style["marker"], lw=style["lw"])
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Distance-based recall")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", alpha=0.4)
    ax.set_title(f"Recall vs. dimension at $N{{=}}500\\mathrm{{k}}$, $k{{=}}40$ (best quality setting)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_recall_speed_pareto(out_path: Path) -> None:
    df = _load_recall_dist()
    if df.empty: print("  [skip] recall_speed_pareto — no data"); return
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.astype({"dim": int, "points": int, "k": int})
    # Use d=3 N=500k k=40 cell, plot recall vs time_ms
    sub = df[(df["dim"] == 3) & (df["points"] == 500_000) & (df["k"] == 40)]
    if "time_ms" not in sub.columns:
        print(f"  [skip] recall_speed_pareto — time_ms not in recall_dist; using mean recall only")
        return plot_recall_controlled(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for be in ("pca_fgc", "faiss", "cuvs_bf", "ggnn"):
        s = sub[sub["be"] == be]
        if s.empty: continue
        style = BACKEND[be]
        ax.scatter(s["recall"], s["time_ms"], label=style["label"],
                   color=style["color"], marker=style["marker"], s=60)
    ax.set_xlabel("Distance-based recall")
    ax.set_ylabel("Wall-clock (ms)")
    ax.set_yscale("log")
    ax.set_title("Recall–speed Pareto (d=3, N=500k, k=40)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_recall_vs_dimension(out_path: Path) -> None:
    df = _load_recall_dist()
    if df.empty: print("  [skip] recall_vs_dimension — no data"); return
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.astype({"dim": int, "points": int, "k": int})
    sub = df[(df["points"] == 100_000) & (df["k"] == 40)]
    if sub.empty:
        sub = df[df["k"] == 40]
    fig, ax = plt.subplots(figsize=(8, 5))
    for be in ("pca_fgc", "faiss", "cuvs_bf", "ggnn"):
        s = sub[sub["be"] == be]
        if s.empty: continue
        agg = s.groupby("dim")["recall"].mean().reset_index().sort_values("dim")
        style = BACKEND[be]
        ax.plot(agg["dim"], agg["recall"], label=style["label"],
                color=style["color"], marker=style["marker"], lw=style["lw"])
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Distance-based recall")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", ls=":", alpha=0.4)
    ax.set_title("Recall vs. dimension (N=100k, k=40)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading from: {PERF}")
    print(f"Writing to:   {OUT}")
    print()
    print("== HGCAL plots ==")
    hgcal = load_hgcal_timing()
    plot_hgcal_dim_scaling_5M(hgcal, OUT / "gpu_fgc_dimensional_scaling_5M_k40_all_algorithms.png")
    plot_hgcal_k_comparison_1M(hgcal, OUT / "gpu_fgc_k_comparison_1M_d2-10_all_algorithms.png")
    plot_hgcal_size_scaling(hgcal, dim=3, out_path=OUT / "gpu_fgc_speedup_d3_all_algorithms.png")
    plot_hgcal_size_scaling(hgcal, dim=8, out_path=OUT / "gpu_fgc_speedup_d8_gpu.png")

    print("\n== Synthetic plots ==")
    synth = load_synth_timing()
    plot_synth_dim_scaling(synth, OUT / "synth_dimensional_scaling_1M_k40_gpu.png")
    plot_synth_size_scaling(synth, dim=3, out_path=OUT / "synth_speedup_d3_gpu.png")
    plot_synth_size_scaling(synth, dim=8, out_path=OUT / "synth_speedup_d8_gpu.png")

    print("\n== CLOVER head-to-head ==")
    plot_clover_headtohead(OUT / "clover_headtohead_d3.png")

    print("\n== Memory ==")
    plot_memory(OUT / "memory_footprint.png")

    print("\n== Recall plots ==")
    plot_recall_exactness(OUT / "recall_exactness_proof.png")
    plot_recall_controlled(OUT / "recall_controlled_comparison.png")
    plot_recall_speed_pareto(OUT / "recall_speed_pareto.png")
    plot_recall_vs_dimension(OUT / "recall_vs_dimension.png")

    print(f"\nDone. Outputs in {OUT}")
    for p in sorted(OUT.glob("*.png")):
        sz = p.stat().st_size
        print(f"  {p.name} — {sz//1024:>5} KB")


if __name__ == "__main__":
    main()
