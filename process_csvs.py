
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    'GGNN': {
        'display_name': 'GGNN',
        'color': '#7570B3',  # Professional purple
        'time_column': 'ggnn_time',
        'marker_symbol': 'cross'
    },
    'FGC': {
        'display_name': 'FGC',
        'color': '#F18F01',  # Professional yellow-orange
        'time_column': 'fgc_time',
        'marker_symbol': 'star'
    }
}

MAX_DATASET_SIZE = 5_000_000  # Cap all analyses at 5M data points


def process_csv(file_path: str, output_path: str, time_cols: list, key_cols: list):
    """
    Loads a CSV, processes duplicates by weighted average, and saves the result.

    Args:
        file_path: Path to the input CSV file.
        output_path: Path to save the cleaned CSV file.
        time_cols: List of time columns to apply weighted average on.
        key_cols: List of columns to group by.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return

    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    if 'dims' in df.columns and 'dimension' not in df.columns:
        df = df.rename(columns={'dims': 'dimension'})
        df.to_csv(file_path, index=False)
        print(f"Renamed 'dims' to 'dimension' in {file_path}")

    # Ensure all necessary columns are present
    required_cols = key_cols + time_cols + ['count']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns in {file_path}. Skipping.")
        return

    # Define aggregation logic
    agg_funs = {}
    for col in time_cols:
        agg_funs[col] = lambda x: np.average(
            x, weights=df.loc[x.index, 'count'])

    agg_funs['count'] = 'sum'

    # Group and aggregate
    df_cleaned = df.groupby(key_cols).agg(agg_funs).reset_index()

    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare unified algorithm performance data."""
    print("Loading performance data...")

    # Load all datasets
    data_files = {
        'FAISS': 'faiss_data_cleaned.csv',
        'SCANN': 'scann_data_cleaned.csv',
        'HNSWLIB': 'hnswlib_data_cleaned.csv',
        'ANNOY': 'annoy_data_cleaned.csv',
        'GGNN': 'ggnn_data_cleaned.csv'
    }

    datasets = {}
    for alg, filename in data_files.items():
        if os.path.exists(filename):
            datasets[alg] = pd.read_csv(filename)
            print(f"Loaded {alg}: {len(datasets[alg])} records")
        else:
            print(f"Warning: {filename} not found, skipping {alg}")

    if not datasets:
        print("No cleaned data files found!")
        return pd.DataFrame()

    # Start with the first available dataset as base
    base_alg = list(datasets.keys())[0]
    unified_data = datasets[base_alg].copy()

    # Standardize column names - ensure 'dimension' column exists
    if 'dims' in unified_data.columns and 'dimension' not in unified_data.columns:
        unified_data = unified_data.rename(columns={'dims': 'dimension'})

    # Merge other algorithms
    for alg, df in datasets.items():
        if alg == base_alg:
            continue

        # Standardize column names
        if 'dims' in df.columns and 'dimension' not in df.columns:
            df = df.rename(columns={'dims': 'dimension'})

        # Get the time columns for this algorithm
        alg_time_col = ALGORITHMS[alg]['time_column']
        merge_cols = ['size', 'k', 'dimension', alg_time_col, 'fgc_time']

        # Only keep columns that exist
        available_cols = [col for col in merge_cols if col in df.columns]

        if len(available_cols) >= 3:  # At least size, k, dimension
            unified_data = pd.merge(
                unified_data,
                df[available_cols],
                on=['size', 'k', 'dimension'],
                how='outer',
                suffixes=('', f'_{alg.lower()}')
            )

    print(f"Loaded unified data: {len(unified_data)} records")
    return unified_data


def plot_fgc_speedup_analysis(data: pd.DataFrame, analysis_type: str, **kwargs) -> go.Figure:
    """Create FGC speedup analysis plot with all algorithms."""

    if analysis_type == 'dimensions':
        size = kwargs.get('size', 1_000_000)
        k = kwargs.get('k', 40)
        max_dimensions = kwargs.get('max_dimensions', 20)
        title = f"FGC Speedup Analysis: {size//1_000_000}M Points, K={k}"

        # Filter data
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
        k = kwargs.get('k', 40)
        title = f"FGC Speedup Analysis: D={dimension}, K={k}, Varying Sizes"

        # Filter data
        filtered_data = data[
            (data['dimension'] == dimension) &
            (data['k'] == k) &
            (data['size'] <= MAX_DATASET_SIZE)
        ].copy().sort_values('size')

        x_col = 'size'
        x_title = "Dataset Size"
        x_range = [0, 5_000_000]

    # Create figure
    fig = go.Figure()

    # Plot all algorithms
    for algorithm in ['FAISS', 'SCANN', 'HNSWLIB', 'ANNOY', 'GGNN']:
        if algorithm not in ALGORITHMS:
            continue

        alg_info = ALGORITHMS[algorithm]
        time_col = alg_info['time_column']

        if time_col not in filtered_data.columns or 'fgc_time' not in filtered_data.columns:
            continue

        # Calculate speedup
        valid_data = filtered_data[
            (filtered_data[time_col].notna()) &
            (filtered_data['fgc_time'].notna()) &
            (filtered_data[time_col] > 0) &
            (filtered_data['fgc_time'] > 0)
        ].copy()

        if valid_data.empty:
            continue

        speedup = valid_data[time_col] / valid_data['fgc_time']

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

    fig.update_yaxes(
        gridcolor='lightgray',
        title="FGC Speedup Factor",
        title_font_size=14
    )

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


def plot_k_comparison_dimensional_analysis(data: pd.DataFrame) -> go.Figure:
    """Create three-panel side-by-side plot for dimensional analysis at 1M points."""
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
            size=1_000_000,
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
        title_text="FGC Dimensional Scaling Analysis: K Comparison (1M Vectors, d=2-10)",
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


def save_figure(fig: go.Figure, filename: str, width: int = 1200, height: int = 600):
    """Save figure with professional settings."""
    fig.write_image(filename, width=width, height=height, scale=2)
    print(f"✓ Saved: {filename}")


def create_plots():
    """Create all required plots after data processing."""
    print("\n" + "="*60)
    print("Creating Performance Analysis Plots")
    print("="*60)

    # Load unified data
    data = load_and_prepare_data()

    if data.empty:
        print("No data available for plotting!")
        return

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 1. d3: K=40, size iterate
    print("\n1. Creating FGC speedup analysis: D=3, K=40, varying sizes...")
    fig1 = plot_fgc_speedup_analysis(data, 'sizes', dimension=3, k=40)
    save_figure(fig1, 'plots/fgc_speedup_d3_all_algorithms.png', 1200, 500)

    # 2. d5: K=40, size iterate
    print("\n2. Creating FGC speedup analysis: D=5, K=40, varying sizes...")
    fig2 = plot_fgc_speedup_analysis(data, 'sizes', dimension=5, k=40)
    save_figure(fig2, 'plots/fgc_speedup_d5_all_algorithms.png', 1200, 500)

    # 3. k comparison: 10, 40, 100 @ dimension iteration, 1M size
    print("\n3. Creating K comparison dimensional analysis: K=10,40,100 @ 1M...")
    fig3 = plot_k_comparison_dimensional_analysis(data)
    save_figure(
        fig3, 'plots/fgc_k_comparison_1M_d2-10_all_algorithms.png', 1800, 600)

    # 4. dimensional scaling, 1M, K=40
    print("\n4. Creating dimensional scaling analysis: 1M, K=40...")
    fig4 = plot_fgc_speedup_analysis(
        data, 'dimensions', size=1_000_000, k=40, max_dimensions=15)
    save_figure(
        fig4, 'plots/fgc_dimensional_scaling_1M_k40_all_algorithms.png', 800, 500)

    print(f"\n" + "="*60)
    print("Plot Generation Complete! Generated files:")
    print("• plots/fgc_speedup_d3_all_algorithms.png")
    print("• plots/fgc_speedup_d5_all_algorithms.png")
    print("• plots/fgc_k_comparison_1M_d2-10_all_algorithms.png")
    print("• plots/fgc_dimensional_scaling_1M_k40_all_algorithms.png")
    print("="*60)


def main():
    """Main function to process all algorithm CSVs."""

    algorithms_to_process = [
        {
            'name': 'GGNN',
            'file': 'ggnn_data.csv',
            'time_cols': ['ggnn_time', 'fgc_time'],
        },
        {
            'name': 'Annoy',
            'file': 'annoy_data.csv',
            'time_cols': ['annoy_time', 'fgc_time'],
        },
        {
            'name': 'FAISS',
            'file': 'faiss_data.csv',
            'time_cols': ['faiss_time', 'fgc_time'],
        },
        {
            'name': 'HNSWLIB',
            'file': 'hnswlib_data.csv',
            'time_cols': ['hnswlib_time', 'fgc_time'],
        },
        {
            'name': 'ScaNN',
            'file': 'scann_data.csv',
            'time_cols': ['scann_time', 'fgc_time'],
        },
    ]

    key_columns = ['size', 'k', 'dimension']

    # In scann_data.csv, the dimension column is named 'dimension' not 'dims'
    # so no special handling is needed if the file is consistent.
    # We will add a check for 'dims' and rename it for compatibility.

    print("Starting CSV processing...")

    for algo in algorithms_to_process:
        file_path = algo['file']
        output_file = file_path.replace('.csv', '_cleaned.csv')
        process_csv(file_path, output_file, algo['time_cols'], key_columns)

    print("\nProcessing complete.")

    # Create plots after processing all data
    create_plots()


if __name__ == "__main__":
    main()
