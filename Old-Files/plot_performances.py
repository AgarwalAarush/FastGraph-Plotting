#!/usr/bin/env python3
"""
Performance Comparison Analysis for Nearest Neighbor Search Algorithms
======================================================================

This module creates professional, publication-ready visualizations comparing
the performance of FGC against state-of-the-art methods including FAISS, ScaNN, HNSWLIB, and Annoy.

The code is designed to be easily extensible for additional algorithms.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any

# Global configuration for algorithms
ALGORITHMS = {
    'FAISS': {
        'display_name': 'FAISS',
        'color': '#1B9E77',  # Professional green
        'time_column': 'faiss_time',
        'marker_symbol': 'circle'
    },
    'SCANN': {
        'display_name': 'ScaNN',
        'color': '#D95F02',  # Professional orange
        'time_column': 'scann_time',
        'marker_symbol': 'triangle-up'
    },
    'HNSWLIB': {
        'display_name': 'HNSWLIB',
        'color': '#2E86AB',  # Professional blue
        'time_column': 'hnswlib_time',
        'marker_symbol': 'diamond'
    },
    'ANNOY': {
        'display_name': 'Annoy',
        'color': '#A23B72',  # Professional magenta
        'time_column': 'annoy_time',
        'marker_symbol': 'square'
    },
    'FGC': {
        'display_name': 'FGC',
        'color': '#F18F01',  # Professional yellow-orange
        'time_column': 'fgc_time',
        'marker_symbol': 'star'
    }
}

BASELINE_ALGORITHM = 'FGC'  # Algorithm to compare others against
MAX_DATASET_SIZE = 5_000_000  # Cap all analyses at 5M data points


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load and prepare unified algorithm performance data.

    Returns:
        Unified DataFrame with all algorithm data
    """
    print("Loading performance data...")

    # Load all datasets
    faiss_data = pd.read_csv(
        '/Users/aarushagarwal/Documents/CMU/Research/Plots/faiss_data_cleaned.csv')
    scann_data = pd.read_csv(
        '/Users/aarushagarwal/Documents/CMU/Research/Plots/scann_data_cleaned.csv')
    hnswlib_data = pd.read_csv(
        '/Users/aarushagarwal/Documents/CMU/Research/Plots/hnswlib_data_cleaned.csv')
    annoy_data = pd.read_csv(
        '/Users/aarushagarwal/Documents/CMU/Research/Plots/annoy_data_cleaned.csv')

    # Standardize column names
    scann_data = scann_data.rename(columns={'dimension': 'dims'})
    annoy_data = annoy_data.rename(columns={'dimension': 'dims'})

    # Create unified dataset by merging all data
    # Start with FAISS data as base
    unified_data = faiss_data.copy()

    # Add ScaNN data
    unified_data = pd.merge(unified_data, scann_data[['size', 'k', 'dims', 'scann_time', 'fgc_time']],
                            left_on=['size', 'k', 'dimension'], right_on=['size', 'k', 'dims'],
                            how='outer', suffixes=('', '_scann'))

    # Add HNSWLIB data
    unified_data = pd.merge(unified_data, hnswlib_data[['size', 'k', 'dimension', 'hnswlib_time', 'fgc_time']],
                            on=['size', 'k', 'dimension'], how='outer', suffixes=('', '_hnswlib'))

    # Add Annoy data
    unified_data = pd.merge(unified_data, annoy_data[['size', 'k', 'dims', 'annoy_time', 'fgc_time']],
                            left_on=['size', 'k', 'dimension'], right_on=['size', 'k', 'dims'],
                            how='outer', suffixes=('', '_annoy'))

    # Clean up and standardize columns
    unified_data = unified_data.drop(
        columns=['dims', 'dims_scann', 'dims_annoy'], errors='ignore')
    unified_data = unified_data.rename(columns={'fgc_time': 'fgc_time_faiss'})

    # Fill missing values with NaN
    time_columns = ['faiss_time', 'scann_time', 'hnswlib_time',
                    'annoy_time', 'fgc_time_faiss', 'fgc_time_scann', 'fgc_time_hnswlib', 'fgc_time_annoy']
    for col in time_columns:
        if col not in unified_data.columns:
            unified_data[col] = np.nan

    print(f"Loaded unified data: {len(unified_data)} records")
    return unified_data


def plot_fgc_speedup_analysis(data: pd.DataFrame, analysis_type: str, custom_title: str = None, y_axis_cap: Optional[int] = None, **kwargs) -> go.Figure:
    """
    Create unified FGC speedup analysis plot with all four algorithms.

    Args:
        data: Unified DataFrame with all algorithm data
        analysis_type: 'dimensions' or 'sizes'
        y_axis_cap: Optional integer to cap the y-axis.
        **kwargs: Additional parameters (size, k, dimension, etc.)

    Returns:
        Plotly Figure object
    """
    if analysis_type == 'dimensions':
        size = kwargs.get('size', 1_000_000)
        k = kwargs.get('k', 40)
        max_dimensions = kwargs.get('max_dimensions', 20)
        title = f"FGC Speedup Analysis: {size//1_000_000}M Points, K={k}, Max Dimensions = {max_dimensions}"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        # Filter data for specific size and k
        filtered_data = data[
            (data['size'] == size) &
            (data['k'] == k) &
            (data['dimension'] <= max_dimensions)
        ].copy().sort_values('dimension')

        x_col = 'dimension'
        x_title = "Number of Dimensions (d)"
        x_range = [1, max_dimensions]

    else:  # sizes
        dimension = kwargs.get('dimension', 3)
        k = kwargs.get('k', 10)
        interval = kwargs.get('interval', 500_000)
        title = f"FGC Speedup Analysis: D={dimension}, Varying Sizes"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        # Filter data for specific dimension and k
        filtered_data = data[
            (data['dimension'] == dimension) &
            (data['k'] == k) &
            (data['size'] <= MAX_DATASET_SIZE)
        ].copy().sort_values('size')

        # Apply interval filtering
        if interval:
            target_sizes = [100_000, 300_000] + \
                list(range(interval, MAX_DATASET_SIZE + 1, interval))
            filtered_data = filtered_data[filtered_data['size'].isin(
                target_sizes)]

        x_col = 'size'
        x_title = "Dataset Size"
        x_range = [0, 5_000_000]

    # Create figure
    fig = go.Figure()

    # Plot all four algorithms
    for algorithm in ['FAISS', 'SCANN', 'HNSWLIB', 'ANNOY']:
        alg_info = ALGORITHMS[algorithm]
        time_col = alg_info['time_column']

        # Determine the correct FGC time column for the algorithm
        fgc_col = f"fgc_time_{algorithm.lower()}"

        if fgc_col not in filtered_data.columns:
            continue

        # Calculate speedup
        valid_data = filtered_data[
            (filtered_data[time_col].notna()) &
            (filtered_data[fgc_col].notna()) &
            (filtered_data[time_col] > 0) &
            (filtered_data[fgc_col] > 0)
        ].copy()

        if valid_data.empty:
            continue

        speedup = valid_data[time_col] / valid_data[fgc_col]

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=valid_data[x_col],
                y=speedup,
                mode='lines+markers',
                name=alg_info['display_name'],
                line=dict(color=alg_info['color'], width=3),
                marker=dict(size=8, symbol=alg_info['marker_symbol']),
                showlegend=True,
                hovertemplate=(
                    f"<b>{alg_info['display_name']}</b><br>"
                    f"{x_title}: %{{x}}<br>"
                    "Speedup: %{{y:.2f}}×<br>"
                    "<extra></extra>"
                )
            )
        )

    # Add reference line at y=1
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[1, 1],
            mode='lines',
            name='No Speedup',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=True,
            hovertemplate="No speedup reference<extra></extra>"
        )
    )

    # Update layout
    fig.update_xaxes(
        range=x_range,
        gridcolor='lightgray',
        title=x_title,
        title_font_size=14
    )

    if analysis_type == 'sizes':
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1_000_000, 2_000_000,
                      3_000_000, 4_000_000, 5_000_000],
            ticktext=['0', '1M', '2M', '3M', '4M', '5M']
        )

    y_axis_config = {
        'gridcolor': 'lightgray',
        'title': "FGC Speedup Factor",
        'title_font_size': 14
    }
    if y_axis_cap:
        y_axis_config['range'] = [0, y_axis_cap]

    fig.update_yaxes(**y_axis_config)

    if custom_title:
        title = custom_title

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font_size=16,
            font_family="Arial"
        ),
        height=500,
        width=800 if analysis_type == 'dimensions' else 1200,
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=80, r=80)
    )

    return fig


def save_figure(fig: go.Figure, filename: str, width: int = 1200, height: int = 600):
    """Save figure with professional settings."""
    fig.write_image(filename, width=width, height=height, scale=2)
    print(f"✓ Saved: {filename}")


def plot_dimensional_analysis_side_by_side(data: pd.DataFrame) -> go.Figure:
    """
    Creates a side-by-side plot for dimensional analysis, showing both
    a standard and a zoomed view.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Standard View (d ≤ 10)",
                        "Zoomed View (d ≤ 15, Y-Axis Capped)"),
        horizontal_spacing=0.06
    )

    # Generate figures to extract traces
    fig_normal = plot_fgc_speedup_analysis(
        data, 'dimensions', custom_title=" ", size=1_000_000, k=40, max_dimensions=10)
    fig_zoomed = plot_fgc_speedup_analysis(
        data, 'dimensions', custom_title=" ", size=1_000_000, k=40, max_dimensions=15, y_axis_cap=50)

    # Add traces from the normal plot to the first subplot
    for trace in fig_normal.data:
        fig.add_trace(trace, row=1, col=1)

    # Add traces from the zoomed plot to the second subplot
    # Set showlegend to False to avoid duplicate legend entries
    for trace in fig_zoomed.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Update axes for a cohesive look
    fig.update_xaxes(title_text="Number of Dimensions (d)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Dimensions (d)", row=1, col=2)
    fig.update_yaxes(title_text="FGC Speedup Factor", row=1, col=1)
    fig.update_yaxes(title_text="FGC Speedup Factor",
                     range=fig_zoomed.layout.yaxis.range, row=1, col=2)

    # Update main layout
    fig.update_layout(
        title_text="FGC Dimensional Scaling Analysis (1M Vectors, K=40)",
        height=600,
        width=1400,
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=150, l=80, r=80)
    )
    return fig


def plot_k_comparison_dimensional_analysis(data: pd.DataFrame) -> go.Figure:
    """
    Creates a three-panel side-by-side plot for dimensional analysis at 5M points,
    comparing k=10, k=40, and k=100 with dimensions varying from 2 to 10.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("K=10", "K=40", "K=100"),
        horizontal_spacing=0.08
    )

    k_values = [10, 40, 100]

    for i, k in enumerate(k_values, 1):
        # Generate individual figure for this k value
        single_fig = plot_fgc_speedup_analysis(
            data, 'dimensions',
            custom_title=" ",
            size=5_000_000,
            k=k,
            max_dimensions=10
        )

        # Add traces from this figure to the appropriate subplot
        for j, trace in enumerate(single_fig.data):
            # Only show legend for the first subplot to avoid duplicates
            trace.showlegend = (i == 1)
            fig.add_trace(trace, row=1, col=i)

    # Update axes for all subplots
    for i in range(1, 4):
        fig.update_xaxes(
            title_text="Number of Dimensions (d)",
            range=[2, 10],
            row=1, col=i
        )
        fig.update_yaxes(
            title_text="FGC Speedup Factor" if i == 1 else "",
            row=1, col=i
        )

    # Update main layout
    fig.update_layout(
        title_text="FGC Dimensional Scaling Analysis: K Comparison (5M Vectors, d=2-10)",
        height=600,
        width=1800,
        template='plotly_white',
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=120, l=80, r=80)
    )
    return fig


def main():
    """Main execution function."""
    print("="*60)
    print("Performance Analysis: FGC vs State-of-the-Art Methods")
    print("="*60)

    # Load unified data
    data = load_and_prepare_data()

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # --- Combined Dimensional Analysis ---
    print("\n--- Generating Combined Dimensional Analysis Plot ---")
    print("1. Creating side-by-side dimensional analysis plot...")
    fig_combined_dim = plot_dimensional_analysis_side_by_side(data)
    save_figure(fig_combined_dim,
                'plots/fgc_dimensional_scaling_analysis.png', width=1200, height=600)

    # --- K Comparison Dimensional Analysis ---
    print("\n--- Generating K Comparison Dimensional Analysis ---")
    print("2. Creating k=10, k=40, k=100 comparison for 5M points, dimensions 2-10...")
    fig_k_comparison = plot_k_comparison_dimensional_analysis(data)
    save_figure(fig_k_comparison,
                'plots/fgc_k_comparison_5M_d2-10.png', width=1800, height=600)

    # --- Standard Analysis ---
    print("\n--- Generating Standard Speedup Plots ---")
    # 3) FGC Speedup Analysis: D=3, Varying Sizes
    print(f"\n3. Creating FGC speedup analysis: D=3, varying sizes...")
    fig3 = plot_fgc_speedup_analysis(
        data, 'sizes', custom_title="Speedup Factor at D=3 and K=40", dimension=3, k=40, interval=500_000)
    save_figure(fig3, 'plots/fgc_speedup_d3_professional.png', 1200, 500)

    # 4) FGC Speedup Analysis: D=5, Varying Sizes
    print(f"\n4. Creating FGC speedup analysis: D=5, varying sizes...")
    fig4 = plot_fgc_speedup_analysis(
        data, 'sizes', custom_title="Speedup Factor at D=5 and K=40", dimension=5, k=40, interval=500_000)
    save_figure(fig4, 'plots/fgc_speedup_d5_professional.png', 1200, 500)

    print(f"\n" + "="*60)
    print("Analysis Complete! Generated FGC speedup visualizations:")
    print("• plots/fgc_dimensional_scaling_analysis.png")
    print("• plots/fgc_k_comparison_5M_d2-10.png")
    print("• plots/fgc_speedup_d3_professional.png")
    print("• plots/fgc_speedup_d5_professional.png")
    print("="*60)


if __name__ == "__main__":
    main()
