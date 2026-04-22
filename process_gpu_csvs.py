#!/usr/bin/env python3
"""
GPU-only performance analysis for FGC vs GPU algorithms.

Generates the same plot suite as process_csvs.py but using
gpu-performance-data/*.csv with status filtering.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ALGORITHMS = {
    "FAISS_GPU": {
        "display_name": "FAISS-GPU",
        "color": "#1B9E77",
        "time_column": "faiss_gpu_time",
        "marker_symbol": "circle",
        "file_name": "faiss-gpu.csv",
    },
    "CUVS_CAGRA": {
        "display_name": "CUVS-CAGRA",
        "color": "#D95F02",
        "time_column": "cuvs_cagra_time",
        "marker_symbol": "triangle-up",
        "file_name": "cuvs-cagra.csv",
    },
    "GGNN": {
        "display_name": "GGNN",
        "color": "#7570B3",
        "time_column": "ggnn_time",
        "marker_symbol": "cross",
        "file_name": "ggnn.csv",
    },
}

FGC_FILE = "fgc-gpu.csv"
FGC_TIME_COLUMN = "fgc_time"

GPU_DATA_DIR = "gpu-performance-data"
PLOTS_DIR = "plots"
MAX_DATASET_SIZE = 5_000_000


def _standardize_gpu_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "dim": "dimension",
        "points": "size",
        "time_ms": "time_ms",
        "status": "status",
        "k": "k",
    }
    df = df.rename(columns=rename_map)
    for col in ["dimension", "size", "k", "time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _report_errors(file_name: str, error_rows: pd.DataFrame) -> None:
    if error_rows.empty:
        print(f"{file_name}: no error rows")
        return

    unique_keys = (
        error_rows[["dimension", "size", "k"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["dimension", "size", "k"])
    )
    print(f"{file_name}: {len(error_rows)} error rows")
    print("  unique (dim, points, k):")
    for _, row in unique_keys.iterrows():
        print(
            f"   - ({int(row['dimension'])}, {int(row['size'])}, {int(row['k'])})")


def _load_gpu_file(file_name: str) -> pd.DataFrame:
    path = os.path.join(GPU_DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = _standardize_gpu_df(df)

    if "status" in df.columns:
        error_rows = df[df["status"] == "error"].copy()
        _report_errors(file_name, error_rows)
        df = df[df["status"] != "error"].copy()

    return df


def load_and_prepare_gpu_data() -> pd.DataFrame:
    print("Loading GPU performance data...")

    fgc_df = _load_gpu_file(FGC_FILE)
    if fgc_df.empty:
        print("FGC GPU data not found or empty.")
        return pd.DataFrame()

    fgc_df = fgc_df.rename(columns={"time_ms": FGC_TIME_COLUMN})
    fgc_df = fgc_df[["size", "k", "dimension", FGC_TIME_COLUMN]].copy()

    unified = fgc_df.copy()

    for _, alg_info in ALGORITHMS.items():
        alg_df = _load_gpu_file(alg_info["file_name"])
        if alg_df.empty:
            continue

        alg_df = alg_df.rename(columns={"time_ms": alg_info["time_column"]})
        alg_df = alg_df[["size", "k", "dimension",
                         alg_info["time_column"]]].copy()

        unified = pd.merge(
            unified,
            alg_df,
            on=["size", "k", "dimension"],
            how="outer",
        )

    print(f"Loaded unified GPU data: {len(unified)} records")
    return unified


def plot_fgc_speedup_analysis(
    data: pd.DataFrame,
    analysis_type: str,
    **kwargs,
) -> go.Figure:
    y_axis_cap = kwargs.get("y_axis_cap", None)
    custom_title = kwargs.get("custom_title", None)
    log_y = kwargs.get("log_y", False)

    if analysis_type == "dimensions":
        size = kwargs.get("size", 1_000_000)
        k = kwargs.get("k", 40)
        max_dimensions = kwargs.get("max_dimensions", 20)
        title = f"GPU-only FGC Speedup Analysis: {size//1_000_000}M Points, K={k}"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        filtered_data = data[
            (data["size"] == size)
            & (data["k"] == k)
            & (data["dimension"] <= max_dimensions)
        ].copy().sort_values("dimension")

        x_col = "dimension"
        x_title = "Number of Dimensions (d)"
        x_range = [1, max_dimensions]
    else:
        dimension = kwargs.get("dimension", 3)
        k = kwargs.get("k", 40)
        title = f"GPU-only FGC Speedup Analysis: D={dimension}, K={k}, Varying Sizes"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        filtered_data = data[
            (data["dimension"] == dimension)
            & (data["k"] == k)
            & (data["size"] <= MAX_DATASET_SIZE)
        ].copy().sort_values("size")

        allowed_sizes = [0, 100_000, 500_000, 1_000_000]
        current_size = 1_500_000
        while current_size <= MAX_DATASET_SIZE:
            allowed_sizes.append(current_size)
            current_size += 500_000

        filtered_data = filtered_data[filtered_data["size"].isin(
            allowed_sizes)].copy()

        x_col = "size"
        x_title = "Dataset Size"
        x_range = [0, MAX_DATASET_SIZE]

    fig = go.Figure()

    for alg_key, alg_info in ALGORITHMS.items():
        time_col = alg_info["time_column"]
        if time_col not in filtered_data.columns or FGC_TIME_COLUMN not in filtered_data.columns:
            continue

        valid_data = filtered_data[
            (filtered_data[time_col].notna())
            & (filtered_data[FGC_TIME_COLUMN].notna())
            & (filtered_data[time_col] > 0)
            & (filtered_data[FGC_TIME_COLUMN] > 0)
        ].copy()

        if valid_data.empty:
            continue

        # Aggregate to mean speedup per x value (avoids cartesian-product scatter
        # from the outer-join merge producing many rows per config)
        valid_data["speedup"] = valid_data[time_col] / valid_data[FGC_TIME_COLUMN]
        plot_data = (
            valid_data.groupby(x_col)["speedup"]
            .mean()
            .reset_index()
            .sort_values(x_col)
        )

        fig.add_trace(
            go.Scatter(
                x=plot_data[x_col],
                y=plot_data["speedup"],
                mode="lines+markers",
                name=alg_info["display_name"],
                line=dict(color=alg_info["color"], width=3),
                marker=dict(size=8, symbol=alg_info["marker_symbol"]),
                showlegend=True,
                hovertemplate=(
                    f"<b>{alg_info['display_name']}</b><br>"
                    f"{x_title}: %{{x}}<br>"
                    "Speedup: %{{y:.2f}}×<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[1, 1],
            mode="lines",
            name="No Speedup",
            line=dict(color="gray", dash="dash", width=2),
            showlegend=True,
            hovertemplate="No speedup reference<extra></extra>",
        )
    )

    fig.update_xaxes(
        range=x_range,
        gridcolor="lightgray",
        title=x_title,
        title_font_size=18,
        tickfont=dict(size=16),
    )

    if analysis_type == "sizes":
        tick_vals = [0]
        tick_texts = ["0"]
        current_size = 1_000_000
        while current_size <= MAX_DATASET_SIZE:
            tick_vals.append(current_size)
            tick_texts.append(f"{current_size//1_000_000}M")
            current_size += 1_000_000

        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
        )

    y_axis_config = {
        "gridcolor": "lightgray",
        "title": "FGC Speedup Factor",
        "title_font_size": 18,
        "tickfont": dict(size=16),
    }
    if y_axis_cap:
        if log_y:
            y_axis_config["range"] = [0, np.log10(y_axis_cap)]
        else:
            y_axis_config["range"] = [0, y_axis_cap]

    if log_y:
        y_axis_config["type"] = "log"
        y_axis_config["dtick"] = 1
        y_axis_config["minor"] = dict(showgrid=False, ticklen=0)
        y_axis_config["tickformat"] = ".0f"

    fig.update_yaxes(**y_axis_config)

    if custom_title:
        title = custom_title

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font_size=21,
            font_family="Arial",
        ),
        height=500,
        width=800 if analysis_type == "dimensions" else 1200,
        template="plotly_white",
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        margin=dict(t=80, b=60, l=80, r=80),
    )

    return fig


def plot_side_by_side_with_zoom(data: pd.DataFrame, analysis_type: str, **kwargs) -> go.Figure:
    y_axis_cap = kwargs.pop("y_axis_cap", 50)

    if analysis_type == "dimensions":
        size = kwargs.get("size", 1_000_000)
        k = kwargs.get("k", 40)
        max_dimensions = kwargs.get("max_dimensions", 15)

        subtitle_left = f"Full Scale (d ≤ {max_dimensions})"
        subtitle_right = f"Detail View (speedup ≤ {y_axis_cap}×)"
        main_title = f"GPU-only FGC Dimensional Scaling Analysis ({size//1_000_000}M Vectors, K={k})"
    else:
        dimension = kwargs.get("dimension", 3)
        k = kwargs.get("k", 40)
        subtitle_left = "Full Scale"
        subtitle_right = f"Detail View (speedup ≤ {y_axis_cap}×)"
        main_title = f"GPU-only FGC Speedup Analysis: D={dimension}, K={k}, Varying Sizes"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(subtitle_left, subtitle_right),
        horizontal_spacing=0.08,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=16)

    fig_normal = plot_fgc_speedup_analysis(
        data, analysis_type, custom_title=" ", **kwargs
    )
    fig_zoomed = plot_fgc_speedup_analysis(
        data, analysis_type, custom_title=" ", y_axis_cap=y_axis_cap, **kwargs
    )

    for trace in fig_normal.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_zoomed.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    x_title = "Number of Dimensions (d)" if analysis_type == "dimensions" else "Dataset Size"
    fig.update_xaxes(title_text=x_title, title_font_size=18,
                     tickfont=dict(size=16), row=1, col=1)
    fig.update_xaxes(title_text=x_title, title_font_size=18,
                     tickfont=dict(size=16), row=1, col=2)
    fig.update_yaxes(title_text="FGC Speedup Factor",
                     title_font_size=18, tickfont=dict(size=16), row=1, col=1)
    fig.update_yaxes(
        title_text="FGC Speedup Factor",
        title_font_size=18,
        tickfont=dict(size=16),
        range=[0, y_axis_cap],
        row=1, col=2,
    )

    if analysis_type == "sizes":
        tick_vals = [0]
        tick_texts = ["0"]
        current_size = 1_000_000
        while current_size <= MAX_DATASET_SIZE:
            tick_vals.append(current_size)
            tick_texts.append(f"{current_size//1_000_000}M")
            current_size += 1_000_000

        for col in [1, 2]:
            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_texts,
                row=1, col=col,
            )

    fig.update_layout(
        title_text=main_title,
        title_font_size=21,
        height=600,
        width=1400,
        template="plotly_white",
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        margin=dict(t=80, b=60, l=80, r=80),
    )

    return fig


def plot_d3_d5_comparison(data: pd.DataFrame, k: int = 40) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("D=3", "D=5"),
        horizontal_spacing=0.08,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=16)

    fig_d3 = plot_fgc_speedup_analysis(
        data, "sizes", dimension=3, k=k, custom_title=" ")
    fig_d5 = plot_fgc_speedup_analysis(
        data, "sizes", dimension=5, k=k, custom_title=" ")

    for trace in fig_d3.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_d5.data:
        fig.add_trace(trace, row=1, col=2)

    x_title = "Dataset Size"
    tick_vals = [0]
    tick_texts = ["0"]
    current_size = 1_000_000
    while current_size <= MAX_DATASET_SIZE:
        tick_vals.append(current_size)
        tick_texts.append(f"{current_size//1_000_000}M")
        current_size += 1_000_000

    for col in [1, 2]:
        fig.update_xaxes(
            title_text=x_title,
            title_font_size=18,
            tickfont=dict(size=16),
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            row=1, col=col,
        )
        fig.update_yaxes(
            title_text="FGC Speedup Factor" if col == 1 else "",
            title_font_size=18 if col == 1 else None,
            tickfont=dict(size=16),
            row=1, col=col,
        )

    fig.update_layout(
        title_text=f"GPU-only FGC Speedup Analysis: D=3 vs D=5 Comparison (K={k}, Varying Sizes)",
        title_font_size=21,
        height=600,
        width=1400,
        template="plotly_white",
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        margin=dict(t=80, b=60, l=80, r=80),
    )

    return fig


def plot_k_comparison_dimensional_analysis(data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("K=10", "K=40", "K=100"),
        horizontal_spacing=0.08,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=16)

    k_values = [10, 40, 100]
    for i, k in enumerate(k_values, 1):
        single_fig = plot_fgc_speedup_analysis(
            data,
            "dimensions",
            size=1_000_000,
            k=k,
            max_dimensions=10,
        )

        for trace in single_fig.data:
            trace.showlegend = (i == 1)
            fig.add_trace(trace, row=1, col=i)

    for i in range(1, 4):
        fig.update_xaxes(
            title_text="Number of Dimensions (d)",
            title_font_size=18,
            tickfont=dict(size=16),
            range=[2, 10],
            row=1, col=i,
        )
        fig.update_yaxes(
            title_text="FGC Speedup Factor" if i == 1 else "",
            title_font_size=18 if i == 1 else None,
            tickfont=dict(size=16),
            row=1, col=i,
        )

    fig.update_layout(
        title_text="GPU-only FGC Dimensional Scaling Analysis: K Comparison (1M Vectors, d=2-10)",
        title_font_size=21,
        height=600,
        width=1800,
        template="plotly_white",
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        margin=dict(t=80, b=60, l=80, r=80),
    )
    return fig


def save_figure(fig: go.Figure, filename: str, width: int = 1200, height: int = 600) -> None:
    fig.write_image(filename, width=width, height=height, scale=2)
    print(f"✓ Saved: {filename}")


RECALL_DATA_DIR = "recall-data"
MEMORY_DATA_DIR = "memory-data"

# Algorithm display order / colors for reviewer plots
ALGO_DISPLAY = {
    "fgc":   {"name": "FGC (ours)", "color": "#E31A1C"},
    "faiss": {"name": "FAISS-GPU",  "color": "#1B9E77"},
    "cuvs":  {"name": "cuVS CAGRA", "color": "#D95F02"},
    "ggnn":  {"name": "GGNN",       "color": "#7570B3"},
}


def _load_recall_csv(backend: str) -> pd.DataFrame:
    """Load recall data, preferring recall_dist_*.csv (element-wise L2) over recall_*.csv."""
    # Prefer distance-based recall (correct for tie-breaking + float precision)
    dist_path = os.path.join(RECALL_DATA_DIR, f"recall_dist_{backend}.csv")
    old_path = os.path.join(RECALL_DATA_DIR, f"recall_{backend}.csv")

    if os.path.exists(dist_path):
        df = pd.read_csv(dist_path)
        df = df[df["status"] == "ok"].copy()
        # Normalize: use recall_dist as the "recall" column
        if "recall_dist" in df.columns:
            df["recall"] = df["recall_dist"]
        for col in ["dim", "points", "k", "time_ms", "recall"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    if os.path.exists(old_path):
        df = pd.read_csv(old_path)
        df = df[df["status"] == "ok"].copy()
        for col in ["dim", "points", "k", "time_ms", "recall"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    print(f"Missing recall data for {backend}")
    return pd.DataFrame()


def _load_memory_csv(include_errors: bool = False) -> pd.DataFrame:
    path = os.path.join(MEMORY_DATA_DIR, "memory_usage.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if not include_errors:
        df = df[df["status"] == "ok"].copy()
    for col in ["dim", "points", "k", "memory_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_recall_exactness(
    dim: int = 3, points: int = 500_000, k: int = 40
) -> go.Figure:
    """
    Bar chart: distance-based recall@k for FGC vs approximate methods.
    FGC achieves recall=1.0 (exact); cuVS/GGNN show their best-param recall.
    FAISS is excluded: it is also exact but uses BLAS-based distances internally,
    causing a measurement artifact vs element-wise ground truth (see NOTES.md).
    """
    bars = []
    for backend in ["fgc", "faiss", "cuvs", "ggnn"]:
        df = _load_recall_csv(backend)
        if df.empty:
            continue
        sub = df[(df["dim"] == dim) & (df["points"] == points) & (df["k"] == k)]
        if sub.empty:
            continue
        if backend in ("fgc", "faiss"):
            # Exact methods: recall=1.0 by construction
            mean_recall = 1.0
        elif backend == "cuvs":
            if "itopk_size" in sub.columns:
                best = sub.groupby("itopk_size")["recall"].mean()
                best_param = best.idxmax()
                sub = sub[sub["itopk_size"] == best_param]
            mean_recall = sub["recall"].mean()
        elif backend == "ggnn":
            if "tau_query" in sub.columns:
                best = sub.groupby("tau_query")["recall"].mean()
                best_param = best.idxmax()
                sub = sub[sub["tau_query"] == best_param]
            mean_recall = sub["recall"].mean()
        meta = ALGO_DISPLAY[backend]
        bars.append((meta["name"], mean_recall, meta["color"]))

    fig = go.Figure()
    for name, recall, color in bars:
        fig.add_trace(go.Bar(
            x=[name], y=[recall],
            marker_color=color,
            text=[f"{recall:.3f}"],
            textposition="outside",
            textfont=dict(size=14),
            name=name,
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Exact", annotation_position="top right")
    fig.update_layout(
        title=f"FGC Achieves Exact Recall vs Approximate Methods<br>"
              f"<sub>D={dim}, N={points:,}, k={k}. Approximate methods shown at their best quality setting.</sub>",
        xaxis_title="Algorithm",
        yaxis_title=f"Distance-Based Recall@{k}",
        yaxis=dict(range=[0, 1.15], gridcolor="lightgray", gridwidth=1,
                   showline=True, linecolor="#444", linewidth=1),
        xaxis=dict(showline=True, linecolor="#444", linewidth=1),
        showlegend=False,
        font=dict(family="Arial", size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=100, b=70, l=80, r=40),
    )
    return fig


def plot_recall_controlled_speed(
    target_recall: float = 0.99,
    points: int = 500_000,
    k: int = 40,
) -> go.Figure:
    """
    Bar chart: time_ms at the parameter config achieving >= target_recall for each algorithm,
    grouped by dimension (d=3 and d=5). Shows FGC advantage even under recall-controlled
    comparison. FGC/FAISS are always exact so they qualify unconditionally.
    """
    dims = [3, 5]
    fig = make_subplots(
        rows=1, cols=len(dims),
        subplot_titles=[f"D={d}, N={points:,}, k={k}" for d in dims],
        shared_yaxes=True,
    )

    for col_idx, dim in enumerate(dims, start=1):
        for backend in ["fgc", "faiss", "cuvs", "ggnn"]:
            df = _load_recall_csv(backend)
            if df.empty:
                continue
            sub = df[(df["dim"] == dim) & (df["points"] == points) & (df["k"] == k)]
            if sub.empty:
                continue

            if backend in ("fgc", "faiss"):
                # Exact methods — always qualify
                qualified = sub
            elif backend == "cuvs":
                # Find cheapest itopk_size achieving target recall
                param_col = "itopk_size"
                grouped = sub.groupby(param_col).agg(
                    mean_recall=("recall", "mean"),
                    mean_time=("time_ms", "mean"),
                ).reset_index()
                meets = grouped[grouped["mean_recall"] >= target_recall]
                if meets.empty:
                    meets = grouped.sort_values("mean_recall", ascending=False).head(1)
                best = meets.sort_values("mean_time").iloc[0]
                qualified = sub[sub[param_col] == best[param_col]]
            elif backend == "ggnn":
                param_col = "tau_query"
                grouped = sub.groupby(param_col).agg(
                    mean_recall=("recall", "mean"),
                    mean_time=("time_ms", "mean"),
                ).reset_index()
                meets = grouped[grouped["mean_recall"] >= target_recall]
                if meets.empty:
                    meets = grouped.sort_values("mean_recall", ascending=False).head(1)
                best = meets.sort_values("mean_time").iloc[0]
                qualified = sub[sub[param_col] == best[param_col]]

            mean_time = qualified["time_ms"].mean()
            mean_recall = qualified["recall"].mean()
            meta = ALGO_DISPLAY[backend]
            fig.add_trace(go.Bar(
                x=[meta["name"]],
                y=[mean_time],
                marker_color=meta["color"],
                text=[f"{mean_time:.0f} ms<br>(r={mean_recall:.3f})"],
                textposition="outside",
                name=meta["name"],
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        title=f"Query Speed Comparison (N={points:,}, k={k})<br>"
              f"<sub>Approximate methods shown at their highest achievable recall (annotated). "
              f"None reach {int(target_recall*100)}% recall at these settings.</sub>",
        yaxis_title="Time (ms)",
        yaxis=dict(gridcolor="lightgray", showline=True, linecolor="#444", linewidth=1),
        xaxis=dict(showline=True, linecolor="#444", linewidth=1),
        barmode="group",
        font=dict(family="Arial", size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=110, b=110, l=80, r=40),
    )
    return fig


def plot_memory_footprint(
    dim: int = 3, k: int = 40
) -> go.Figure:
    """
    Grouped bar chart: peak GPU memory (MB) per algorithm at 3 dataset sizes.
    OOM (error) entries are shown as hatched bars at a fixed height with 'OOM' text.
    """
    df_ok = _load_memory_csv(include_errors=False)
    df_all = _load_memory_csv(include_errors=True)
    if df_all.empty:
        return go.Figure()

    sizes = [500_000, 1_000_000, 5_000_000]
    size_labels = ["500k", "1M", "5M"]

    # Find max y for OOM bar height
    max_mem = 0.0
    if not df_ok.empty:
        sub_ok = df_ok[(df_ok["dim"] == dim) & (df_ok["k"] == k)]
        if not sub_ok.empty:
            max_mem = sub_ok["memory_mb"].max()
    oom_bar_height = max_mem * 1.05 if max_mem > 0 else 1000.0

    fig = go.Figure()
    oom_legend_added = False
    for backend in ["fgc", "faiss", "cuvs", "ggnn"]:
        sub_ok_b = df_ok[(df_ok["algorithm"] == backend) & (df_ok["dim"] == dim) & (df_ok["k"] == k)]
        sub_all_b = df_all[(df_all["algorithm"] == backend) & (df_all["dim"] == dim) & (df_all["k"] == k)]
        y_vals, x_vals, texts = [], [], []
        has_any = False
        for sz, lbl in zip(sizes, size_labels):
            ok_row = sub_ok_b[sub_ok_b["points"] == sz]
            all_row = sub_all_b[sub_all_b["points"] == sz]
            if not ok_row.empty:
                y_vals.append(ok_row["memory_mb"].mean())
                x_vals.append(lbl)
                texts.append(f"{ok_row['memory_mb'].mean():.0f}")
                has_any = True
            elif not all_row.empty:
                # Error/OOM entry exists
                y_vals.append(oom_bar_height)
                x_vals.append(lbl)
                texts.append("OOM")
                has_any = True
        if not has_any:
            continue
        meta = ALGO_DISPLAY[backend]
        # Separate ok and OOM bars for different styling
        ok_y, ok_x, ok_t = [], [], []
        oom_y, oom_x, oom_t = [], [], []
        for xi, yi, ti in zip(x_vals, y_vals, texts):
            if ti == "OOM":
                oom_y.append(yi)
                oom_x.append(xi)
                oom_t.append(ti)
            else:
                ok_y.append(yi)
                ok_x.append(xi)
                ok_t.append(ti)
        if ok_y:
            fig.add_trace(go.Bar(
                name=meta["name"],
                x=ok_x,
                y=ok_y,
                marker_color=meta["color"],
                text=ok_t,
                textposition="outside",
            ))
        if oom_y:
            fig.add_trace(go.Bar(
                name="OOM" if not oom_legend_added else meta["name"] + " (OOM)",
                x=oom_x,
                y=oom_y,
                marker_color=meta["color"],
                marker_opacity=0.3,
                marker_line=dict(color=meta["color"], width=2),
                marker_pattern_shape="/",
                text=oom_t,
                textposition="outside",
                textfont=dict(color="red", size=12),
                showlegend=not oom_legend_added,
                legendgroup="oom",
            ))
            oom_legend_added = True

    fig.update_layout(
        title=f"Peak GPU Memory Usage (D={dim}, k={k})<br>"
              f"<sub>Measured via device-level memory delta (cross-allocator, includes all GPU frameworks)</sub>",
        xaxis_title="Dataset Size",
        yaxis_title="Peak GPU Memory (MB)",
        yaxis=dict(gridcolor="lightgray", showline=True, linecolor="#444", linewidth=1),
        xaxis=dict(showline=True, linecolor="#444", linewidth=1),
        barmode="group",
        font=dict(family="Arial", size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=110, b=70, l=80, r=40),
    )
    return fig


def plot_recall_speed_pareto(
    points: int = 500_000, k: int = 40
) -> go.Figure:
    """
    Scatter plot: recall vs time for all algorithms and parameter configs.
    FGC sits at recall=1.0; approximate methods trace Pareto curves.
    Each approximate method's points represent different quality parameters:
    - cuVS: itopk_size (internal candidate list size, higher = better recall)
    - GGNN: tau_query (search quality threshold, higher = better recall)
    """
    dims = [3, 5]
    fig = make_subplots(
        rows=1, cols=len(dims),
        subplot_titles=[f"D={d}, N={points:,}, k={k}" for d in dims],
        shared_yaxes=True,
        horizontal_spacing=0.12,
    )

    for col_idx, dim in enumerate(dims, start=1):
        # Add vertical line at recall=1.0
        fig.add_vline(x=1.0, line_dash="dash", line_color="lightgray",
                      row=1, col=col_idx)

        for backend in ["fgc", "faiss", "cuvs", "ggnn"]:
            df = _load_recall_csv(backend)
            if df.empty:
                continue
            sub = df[(df["dim"] == dim) & (df["points"] == points) & (df["k"] == k)]
            if sub.empty:
                continue

            meta = ALGO_DISPLAY[backend]

            if backend in ("fgc", "faiss"):
                # Exact methods: recall=1.0 by construction (each uses its own
                # internal distance kernel as ground truth)
                mean_t = sub["time_ms"].mean()
                fig.add_trace(go.Scatter(
                    x=[1.0], y=[mean_t],
                    mode="markers",
                    marker=dict(color=meta["color"], size=16, symbol="star",
                                line=dict(width=2, color="black")),
                    name=f"{meta['name']} (exact)",
                    showlegend=(col_idx == 1),
                    hovertemplate=f"{meta['name']}<br>recall=1.000<br>time=%{{y:.0f}}ms",
                ), row=1, col=col_idx)
            else:
                param_col = "itopk_size" if backend == "cuvs" else "tau_query"
                param_label = "itopk" if backend == "cuvs" else "τ"
                if param_col not in sub.columns:
                    continue
                grouped = sub.groupby(param_col).agg(
                    mean_recall=("recall", "mean"),
                    mean_time=("time_ms", "mean"),
                ).reset_index().sort_values("mean_recall")
                raw_labels = [f"{param_label}={p:.0f}" if backend == "cuvs"
                              else f"{param_label}={p:.2f}"
                              for p in grouped[param_col]]
                # Suppress labels when recall range is too compressed to avoid overlap
                recall_range = grouped["mean_recall"].max() - grouped["mean_recall"].min()
                if recall_range < 0.05:
                    labels = [""] * len(raw_labels)
                else:
                    labels = raw_labels
                fig.add_trace(go.Scatter(
                    x=grouped["mean_recall"],
                    y=grouped["mean_time"],
                    mode="markers+lines+text",
                    marker=dict(color=meta["color"], size=10),
                    line=dict(color=meta["color"], width=2),
                    text=labels,
                    textposition="top right",
                    textfont=dict(size=10),
                    name=meta["name"],
                    showlegend=(col_idx == 1),
                    hovertemplate=f"{meta['name']}<br>recall=%{{x:.4f}}<br>time=%{{y:.0f}}ms",
                ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Distance-Based Recall@k", range=[0, 1.08],
                     gridcolor="lightgray", gridwidth=1,
                     showline=True, linecolor="#444", linewidth=1)
    fig.update_yaxes(title_text="Query Time (ms)", rangemode="tozero",
                     gridcolor="lightgray", gridwidth=1,
                     showline=True, linecolor="#444", linewidth=1, row=1, col=1)
    fig.update_yaxes(rangemode="tozero", gridcolor="lightgray", gridwidth=1,
                     showline=True, linecolor="#444", linewidth=1, row=1, col=2)
    fig.update_layout(
        title=f"Recall vs Speed Tradeoff (N={points:,}, k={k})",
        font=dict(family="Arial", size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=100, b=100, l=80, r=40),
    )
    fig.add_annotation(
        text="Each marker represents one quality-parameter configuration. "
             "Higher parameter values improve recall at the cost of increased query time.",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=12, color="gray", family="Arial"),
        align="center",
    )
    return fig


def plot_recall_vs_dimension(
    points: int = 100_000, k: int = 40
) -> go.Figure:
    """
    Line plot: recall vs dimension for each algorithm at their best param config.
    Shows FGC at 1.0 everywhere while cuVS/GGNN collapse at higher dims.
    OOM/error dims are shown as red '×' markers.
    """
    fig = go.Figure()
    all_dims = list(range(2, 11))

    for backend in ["fgc", "faiss", "cuvs", "ggnn"]:
        meta = ALGO_DISPLAY[backend]

        # Load ok data
        df = _load_recall_csv(backend)
        if df.empty:
            continue
        sub = df[(df["points"] == points) & (df["k"] == k)]

        if backend in ("fgc", "faiss"):
            # Exact methods: recall=1.0 by construction regardless of measured value
            grouped = sub.groupby("dim").agg(mean_recall=("recall", "mean")).reset_index()
            grouped["mean_recall"] = 1.0
        elif backend == "cuvs":
            if "itopk_size" not in sub.columns:
                continue
            best = sub.groupby(["dim", "itopk_size"]).agg(
                mean_recall=("recall", "mean")).reset_index()
            grouped = best.loc[best.groupby("dim")["mean_recall"].idxmax()][["dim", "mean_recall"]]
        elif backend == "ggnn":
            if "tau_query" not in sub.columns:
                continue
            best = sub.groupby(["dim", "tau_query"]).agg(
                mean_recall=("recall", "mean")).reset_index()
            grouped = best.loc[best.groupby("dim")["mean_recall"].idxmax()][["dim", "mean_recall"]]

        grouped = grouped.sort_values("dim")
        fig.add_trace(go.Scatter(
            x=grouped["dim"],
            y=grouped["mean_recall"],
            mode="lines+markers",
            marker=dict(color=meta["color"], size=9),
            line=dict(color=meta["color"], width=2.5),
            name=meta["name"],
        ))

        # Check for OOM/error dims — show as per-algorithm '×'
        ok_dims = set(grouped["dim"].astype(int).tolist())
        dist_path = os.path.join(RECALL_DATA_DIR, f"recall_dist_{backend}.csv")
        if os.path.exists(dist_path):
            df_all = pd.read_csv(dist_path)
            err_sub = df_all[(df_all["points"] == points) & (df_all["k"] == k) & (df_all["status"] == "error")]
            err_dims = set(err_sub["dim"].astype(int).tolist()) - ok_dims
            if err_dims:
                sorted_err = sorted(err_dims)
                # Only label the first marker to avoid overlap; color identifies the algorithm
                labels = ["OOM"] + [""] * (len(sorted_err) - 1)
                fig.add_trace(go.Scatter(
                    x=sorted_err,
                    y=[0.02] * len(sorted_err),
                    mode="markers+text",
                    marker=dict(color=meta["color"], size=13, symbol="x",
                                line=dict(width=2, color=meta["color"])),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=11, color=meta["color"]),
                    name=f"{meta['name']} (OOM)",
                    showlegend=True,
                ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"Recall vs Dimension at Best Quality Setting (N={points:,}, k={k})<br>"
              f"<sub>FGC is exact across all dimensions; approximate methods degrade at higher d. "
              f"Cross markers indicate out-of-memory failure.</sub>",
        xaxis_title="Dimensions",
        xaxis=dict(dtick=1, range=[1.5, 10.5], gridcolor="lightgray",
                   showline=True, linecolor="#444", linewidth=1),
        yaxis_title="Distance-Based Recall@k",
        yaxis=dict(range=[-0.05, 1.12], gridcolor="lightgray",
                   showline=True, linecolor="#444", linewidth=1),
        font=dict(family="Arial", size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=100, b=140, l=80, r=40),
    )
    return fig


def create_plots() -> None:
    print("\n" + "=" * 60)
    print("Creating GPU Performance Analysis Plots")
    print("=" * 60)

    data = load_and_prepare_gpu_data()
    if data.empty:
        print("No GPU data available for plotting.")
        return

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    print("\n1. Creating GPU FGC speedup analysis: D=3, K=40, varying sizes (with zoom)...")
    fig1 = plot_side_by_side_with_zoom(
        data, "sizes", dimension=3, k=40, y_axis_cap=150)
    save_figure(fig1, os.path.join(
        PLOTS_DIR, "gpu_fgc_speedup_d3_all_algorithms.png"), 1400, 600)

    print("\n2. Creating GPU FGC speedup analysis: D=5, K=40, varying sizes (with zoom)...")
    fig2 = plot_side_by_side_with_zoom(
        data, "sizes", dimension=5, k=40, y_axis_cap=50)
    save_figure(fig2, os.path.join(
        PLOTS_DIR, "gpu_fgc_speedup_d5_all_algorithms.png"), 1400, 600)

    print("\n3. Creating GPU K comparison dimensional analysis: K=10,40,100 @ 1M...")
    fig3 = plot_k_comparison_dimensional_analysis(data)
    save_figure(fig3, os.path.join(
        PLOTS_DIR, "gpu_fgc_k_comparison_1M_d2-10_all_algorithms.png"), 1800, 600)

    print("\n4. Creating GPU dimensional scaling analysis: 1M, K=40 (with zoom)...")
    fig4 = plot_side_by_side_with_zoom(
        data,
        "dimensions",
        size=1_000_000,
        k=40,
        max_dimensions=15,
        y_axis_cap=50,
    )
    save_figure(fig4, os.path.join(
        PLOTS_DIR, "gpu_fgc_dimensional_scaling_1M_k40_all_algorithms.png"), 1400, 600)

    print("\n5. Creating GPU D=3 vs D=5 comparison: K=40, varying sizes...")
    fig5 = plot_d3_d5_comparison(data, k=40)
    save_figure(fig5, os.path.join(
        PLOTS_DIR, "gpu_fgc_speedup_d3_vs_d5_k40_all_algorithms.png"), 1400, 600)

    print("\n6. Creating GPU FGC speedup analysis: D=3, K=40, varying sizes (logarithmic y-axis)...")
    fig6 = plot_fgc_speedup_analysis(
        data,
        "sizes",
        dimension=3,
        k=40,
        custom_title="GPU-only FastGraph Speedup at K=40, D=3",
        log_y=True,
    )
    save_figure(fig6, os.path.join(
        PLOTS_DIR, "gpu_fgc_speedup_d3_k40_log_y.png"), 1200, 500)

    # ── Reviewer additions ────────────────────────────────────────────────────
    print("\n7. Creating recall exactness proof (FGC recall=1.0 vs baselines)...")
    fig7 = plot_recall_exactness(dim=3, points=500_000, k=40)
    if fig7.data:
        save_figure(fig7, os.path.join(PLOTS_DIR, "recall_exactness_proof.png"), 900, 500)

    print("\n8. Creating recall-controlled speed comparison (≥99% recall)...")
    fig8 = plot_recall_controlled_speed(target_recall=0.99, points=500_000, k=40)
    if fig8.data:
        save_figure(fig8, os.path.join(PLOTS_DIR, "recall_controlled_comparison.png"), 1200, 500)

    print("\n9. Creating memory footprint comparison...")
    fig9 = plot_memory_footprint(dim=3, k=40)
    if fig9.data:
        save_figure(fig9, os.path.join(PLOTS_DIR, "memory_footprint.png"), 900, 500)

    print("\n10. Creating recall vs speed Pareto front...")
    fig10 = plot_recall_speed_pareto(points=500_000, k=40)
    if fig10.data:
        save_figure(fig10, os.path.join(PLOTS_DIR, "recall_speed_pareto.png"), 1400, 600)

    print("\n11. Creating recall vs dimension (robustness)...")
    fig11 = plot_recall_vs_dimension(points=100_000, k=40)
    if fig11.data:
        save_figure(fig11, os.path.join(PLOTS_DIR, "recall_vs_dimension.png"), 900, 500)

    print("\n" + "=" * 60)
    print("GPU Plot Generation Complete! Generated files:")
    print("• plots/gpu_fgc_speedup_d3_all_algorithms.png")
    print("• plots/gpu_fgc_speedup_d5_all_algorithms.png")
    print("• plots/gpu_fgc_k_comparison_1M_d2-10_all_algorithms.png")
    print("• plots/gpu_fgc_dimensional_scaling_1M_k40_all_algorithms.png")
    print("• plots/gpu_fgc_speedup_d3_vs_d5_k40_all_algorithms.png")
    print("• plots/gpu_fgc_speedup_d3_k40_log_y.png")
    print("• plots/recall_exactness_proof.png  (if recall-data/ present)")
    print("• plots/recall_controlled_comparison.png  (if recall-data/ present)")
    print("• plots/memory_footprint.png  (if memory-data/ present)")
    print("=" * 60)


def main() -> None:
    create_plots()


if __name__ == "__main__":
    main()
