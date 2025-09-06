#!/usr/bin/env python3
"""
Performance comparison plots between FAISS, SCANN, and FGC algorithms.
Creates various visualizations comparing execution times and speedups.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

def load_and_prepare_data():
    """Load and prepare the data from CSV files."""
    print("Loading data...")
    
    # Load FAISS data
    faiss_data = pd.read_csv('/Users/aarushagarwal/Documents/CMU/Research/Plots/faiss_data.csv')
    print(f"Loaded FAISS data: {len(faiss_data)} rows")
    
    # Load SCANN data 
    scann_data = pd.read_csv('/Users/aarushagarwal/Documents/CMU/Research/Plots/scann_data_impr.csv')
    print(f"Loaded SCANN data: {len(scann_data)} rows")
    
    # Keep original column names and use them consistently
    # faiss_data has 'dimension', scann_data has 'dimension' -> rename scann to match
    scann_data = scann_data.rename(columns={'dimension': 'dims'})
    
    print("Data loaded successfully!")
    return faiss_data, scann_data

def plot_fixed_dims_varying_size(faiss_data, scann_data, dims, k_values, plot_title_suffix=""):
    """Plot performance at fixed dimensions, varying sizes."""
    print(f"Creating plots for {dims} dimensions...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'k={k}' for k in k_values],
        shared_yaxes=True,
        horizontal_spacing=0.1
    )
    
    colors = {'faiss': '#1f77b4', 'fgc': '#ff7f0e', 'scann': '#2ca02c'}
    
    for idx, k in enumerate(k_values, 1):
        # Filter FAISS data
        faiss_subset = faiss_data[
            (faiss_data['dimension'] == dims) & 
            (faiss_data['k'] == k) & 
            (faiss_data['count'] >= 1)  # Filter out failed runs
        ].copy().sort_values('size')  # Sort by size for proper ordering
        
        # Filter SCANN data
        scann_subset = scann_data[
            (scann_data['dims'] == dims) & 
            (scann_data['k'] == k)
        ].copy().sort_values('size')  # Sort by size for proper ordering
        
        if not faiss_subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=faiss_subset['size'],
                    y=faiss_subset['faiss_time'],
                    mode='lines+markers',
                    name='FAISS',
                    line=dict(color=colors['faiss'], width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 1)
                ),
                row=1, col=idx
            )
            
            fig.add_trace(
                go.Scatter(
                    x=faiss_subset['size'],
                    y=faiss_subset['fgc_time'],
                    mode='lines+markers',
                    name='FGC',
                    line=dict(color=colors['fgc'], width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 1)
                ),
                row=1, col=idx
            )
        
        if not scann_subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=scann_subset['size'],
                    y=scann_subset['scann_time'],
                    mode='lines+markers',
                    name='SCANN',
                    line=dict(color=colors['scann'], width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 1)
                ),
                row=1, col=idx
            )
    
    fig.update_xaxes(
        title_text="Dataset Size",
        type="log",
        tickformat=".0s"
    )
    fig.update_yaxes(
        title_text="Time (ms)",
        type="log"
    )
    
    fig.update_layout(
        title=f'Performance Comparison - {dims}D{plot_title_suffix}',
        height=500,
        width=1000,
        template='plotly_white'
    )
    
    filename = f'performance_{dims}d_varying_size.png'
    fig.write_image(filename, width=1000, height=500)
    print(f"Saved {filename}")
    
    return fig

def plot_speedups_varying_dimensions(faiss_data, scann_data, sizes, k_values):
    """Plot speedups over dimensions for fixed sizes."""
    print("Creating speedup plots varying dimensions...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Size: {size//1000}K, k={k}' for size in sizes for k in k_values],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    plot_idx = 1
    for size in sizes:
        for k in k_values:
            row = 1 if plot_idx <= 2 else 2
            col = 1 if plot_idx % 2 == 1 else 2
            
            # FAISS vs FGC speedups - limit to 20 dimensions
            faiss_subset = faiss_data[
                (faiss_data['size'] == size) & 
                (faiss_data['k'] == k) & 
                (faiss_data['count'] >= 1) &
                (faiss_data['faiss_time'] > 0) &
                (faiss_data['dimension'] <= 20)  # Limit to 20 dimensions
            ].copy().sort_values('dimension')  # Sort by dimension for proper ordering
            
            if not faiss_subset.empty:
                faiss_speedup = faiss_subset['faiss_time'] / faiss_subset['fgc_time']
                
                fig.add_trace(
                    go.Scatter(
                        x=faiss_subset['dimension'],
                        y=faiss_speedup,
                        mode='lines+markers',
                        name='FGC vs FAISS',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6),
                        showlegend=(plot_idx == 1)
                    ),
                    row=row, col=col
                )
            
            # SCANN vs FGC speedups - limit to 20 dimensions
            scann_subset = scann_data[
                (scann_data['size'] == size) & 
                (scann_data['k'] == k) &
                (scann_data['dims'] <= 20)  # Limit to 20 dimensions
            ].copy().sort_values('dims')  # Sort by dimension for proper ordering
            
            if not scann_subset.empty:
                scann_speedup = scann_subset['scann_time'] / scann_subset['fgc_time']
                
                fig.add_trace(
                    go.Scatter(
                        x=scann_subset['dims'],
                        y=scann_speedup,
                        mode='lines+markers',
                        name='FGC vs SCANN',
                        line=dict(color='#ff7f0e', width=2),
                        marker=dict(size=6),
                        showlegend=(plot_idx == 1)
                    ),
                    row=row, col=col
                )
            
            # Add horizontal line at y=1 for reference - limit to 20 dimensions
            dims_range = list(range(1, 21))  # Changed to 21 to include 20
            fig.add_trace(
                go.Scatter(
                    x=dims_range,
                    y=[1] * len(dims_range),
                    mode='lines',
                    name='No speedup',
                    line=dict(color='gray', dash='dash', width=1),
                    showlegend=(plot_idx == 1)
                ),
                row=row, col=col
            )
            
            plot_idx += 1
    
    fig.update_xaxes(title_text="Dimensions", range=[1, 20])  # Set x-axis range to 1-20
    fig.update_yaxes(title_text="Speedup (x times faster)")
    
    fig.update_layout(
        title='FGC Speedups vs FAISS and SCANN Across Dimensions',
        height=800,
        width=1200,
        template='plotly_white'
    )
    
    filename = 'speedups_varying_dimensions.png'
    fig.write_image(filename, width=1200, height=800)
    print(f"Saved {filename}")
    
    return fig

def plot_speedups_side_by_side(faiss_data, scann_data, k_values):
    """Create side-by-side speedup plots for different k values."""
    print("Creating side-by-side speedup comparison plots...")
    
    fig = make_subplots(
        rows=1, cols=len(k_values),
        subplot_titles=[f'k = {k}' for k in k_values],
        shared_yaxes=True,
        horizontal_spacing=0.08
    )
    
    for idx, k in enumerate(k_values, 1):
        # FAISS vs FGC speedups
        faiss_subset = faiss_data[
            (faiss_data['k'] == k) & 
            (faiss_data['count'] >= 1) &
            (faiss_data['faiss_time'] > 0)
        ].copy().sort_values('size')  # Sort by size for proper ordering
        
        if not faiss_subset.empty:
            # Group by dimension and calculate mean speedup
            faiss_speedups = faiss_subset.groupby(['dimension', 'size']).apply(
                lambda x: (x['faiss_time'] / x['fgc_time']).mean(), include_groups=False
            ).reset_index(name='speedup')
            faiss_speedups = faiss_speedups.sort_values('size')  # Ensure proper ordering
            
            # Create size categories for better visualization
            faiss_speedups['size_category'] = faiss_speedups['size'].apply(
                lambda x: f"{x//1000}K" if x < 1000000 else f"{x//1000000}M"
            )
            
            # Plot different size categories with lines
            for size in sorted(faiss_speedups['size'].unique()):
                size_data = faiss_speedups[faiss_speedups['size'] == size].sort_values('dimension')
                if not size_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=size_data['size'],
                            y=size_data['speedup'],
                            mode='lines+markers',
                            name=f'FGC vs FAISS',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6, opacity=0.7),
                            showlegend=(idx == 1)
                        ),
                        row=1, col=idx
                    )
        
        # SCANN vs FGC speedups
        scann_subset = scann_data[scann_data['k'] == k].copy().sort_values('size')  # Sort by size
        
        if not scann_subset.empty:
            scann_speedups = scann_subset.groupby(['dims', 'size']).apply(
                lambda x: (x['scann_time'] / x['fgc_time']).mean(), include_groups=False
            ).reset_index(name='speedup')
            scann_speedups = scann_speedups.sort_values('size')  # Ensure proper ordering
            
            for size in sorted(scann_speedups['size'].unique()):
                size_data = scann_speedups[scann_speedups['size'] == size].sort_values('dims')
                if not size_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=size_data['size'],
                            y=size_data['speedup'],
                            mode='lines+markers',
                            name=f'FGC vs SCANN',
                            line=dict(color='orange', width=2),
                            marker=dict(size=6, opacity=0.7),
                            showlegend=(idx == 1)
                        ),
                        row=1, col=idx
                    )
        
        # Add reference line at y=1
        x_range = [1000, 5500000]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[1, 1],
                mode='lines',
                name='No speedup',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=(idx == 1)
            ),
            row=1, col=idx
        )
    
    fig.update_xaxes(
        title_text="Dataset Size",
        type="log",
        tickformat=".0s"
    )
    fig.update_yaxes(
        title_text="Speedup (x times faster)",
        type="log"
    )
    
    fig.update_layout(
        title='FGC Speedups Comparison: Constant Dimension, Varying Size',
        height=600,
        width=1400,
        template='plotly_white'
    )
    
    filename = 'speedups_side_by_side_comparison.png'
    fig.write_image(filename, width=1400, height=600)
    print(f"Saved {filename}")
    
    return fig

def main():
    """Main function to generate all plots."""
    print("Starting performance comparison plotting...")
    
    # Load data
    faiss_data, scann_data = load_and_prepare_data()
    
    # 1) At 3 dims, iterate over sizes for k=10, 40
    print("\n1. Creating plots for 3 dimensions...")
    plot_fixed_dims_varying_size(faiss_data, scann_data, 3, [10, 40])
    
    # 2) At 5 dims, iterate over sizes for k=10, 40  
    print("\n2. Creating plots for 5 dimensions...")
    plot_fixed_dims_varying_size(faiss_data, scann_data, 5, [10, 40])
    
    # 3) At sizes 100k and 1M, plot speedups over dimensions for k=10, 40
    print("\n3. Creating speedup plots varying dimensions...")
    plot_speedups_varying_dimensions(faiss_data, scann_data, [100000, 1000000], [10, 40])
    
    # 4) Side-by-side speedup comparison plots for k=10, 40, 100
    print("\n4. Creating side-by-side speedup comparison plots...")
    plot_speedups_side_by_side(faiss_data, scann_data, [10, 40, 100])
    
    print("\nAll plots generated successfully!")
    print("Generated files:")
    for file in ['performance_3d_varying_size.png', 
                 'performance_5d_varying_size.png',
                 'speedups_varying_dimensions.png',
                 'speedups_side_by_side_comparison.png']:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")

if __name__ == "__main__":
    main()