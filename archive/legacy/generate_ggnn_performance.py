#!/usr/bin/env python3
"""
GGNN vs FastGraphCompute (FGC) Performance Benchmarking Tool
Continuously benchmarks and averages performance data, saving results to CSV.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from fastgraphcompute import binned_select_knn
import ggnn

# Configuration - Set to 'size' or 'dimension' to determine what to vary
TEST_MODE = 'dimension'  # 'size' or 'dimension'

# Fixed parameters
SEED = 42
K = 40

# GGNN parameters
K_BUILD = 24
TAU_BUILD = 0.5
TAU_QUERY = 0.64
MAX_ITERATIONS = 400
DISTANCE_MEASURE = ggnn.DistanceMeasure.Euclidean

# Test ranges
SIZE_RANGE = list(range(100000, 1000000, 100000)) + list(range(1000000, 10000001, 500000))
DIMENSION_RANGE = list(range(2, 11))  # 2 to 50
K_RANGE = [10, 40, 100]  # K values to test
FIXED_SIZE = 5000000  # Used when testing dimensions
FIXED_DIMENSION = 5  # Used when testing sizes

SIZE_MODE_DIM_RANGE = [3, 5]
DIM_MODE_SIZE_RANGE = [1000000]

# Additional iteration options
ITERATE_OVER_K = True  # Set to True to iterate over K values
ITERATE_DIMS_IN_SIZE_MODE = True  # Set to True to iterate over dimensions in size mode
ITERATE_SIZES_IN_DIM_MODE = False  # Set to True to iterate over sizes in dimension mode

# CSV file paths
SIZE_CSV = 'ggnn_data_sizes.csv'
DIM_CSV = 'ggnn_data_dims.csv'

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set GGNN log level (reduce verbosity)
ggnn.set_log_level(1)


def generate_data(n_points, dimensions, seed_offset=0):
    """Generate random data with fixed seed."""
    torch.manual_seed(SEED + seed_offset)
    return torch.rand((n_points, dimensions), dtype=torch.float32, device='cuda')


def time_ggnn(data, k):
    """Time GGNN KNN search."""
    try:
        # Initialize GGNN
        my_ggnn = ggnn.GGNN()
        my_ggnn.set_base(data)
        my_ggnn.set_return_results_on_gpu(True)

        start_time = time.time()

        # Build the graph
        my_ggnn.build(k_build=K_BUILD, tau_build=TAU_BUILD, measure=DISTANCE_MEASURE)

        # Query (using same data as query points)
        indices, dists = my_ggnn.query(data, k, TAU_QUERY, MAX_ITERATIONS, DISTANCE_MEASURE)

        torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) * 1000  # Convert to ms
    except Exception as e:
        print(f"GGNN error: {e}")
        return None


def time_fgc(data, k):
    """Time FGC KNN search."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Ensure data is contiguous
        coordinates = data.contiguous()
        row_splits = torch.tensor([0, len(data)], dtype=torch.int64, device=device)

        # Warm up GPU if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        indices, distances = binned_select_knn(
            k, coordinates, row_splits, direction=None, n_bins=None)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) * 1000  # Convert to ms
    except Exception as e:
        print(f"FGC error: {e}")
        return None


def load_or_create_dataframe(csv_path, test_mode):
    """Load existing CSV or create new DataFrame."""
    if os.path.exists(csv_path):
        print(f"Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Ensure all required columns exist
        required_cols = ['k', 'ggnn_time', 'fgc_time', 'count']
        if test_mode == 'size':
            required_cols.insert(0, 'size')
        else:
            required_cols.insert(0, 'dimension')

        for col in required_cols:
            if col not in df.columns:
                if col == 'k':
                    df[col] = K  # Default to current K value
                elif col == 'count':
                    df[col] = 1
                else:
                    df[col] = 0.0
    else:
        print(f"Creating new DataFrame for {csv_path}")
        if test_mode == 'size':
            df = pd.DataFrame(columns=['size', 'k', 'ggnn_time', 'fgc_time', 'count'])
        else:
            df = pd.DataFrame(columns=['dimension', 'k', 'ggnn_time', 'fgc_time', 'count'])

    return df


def update_average(prev_avg, count, new_value):
    """Update average using count-based incremental averaging."""
    if new_value is None:
        return prev_avg, count
    return (prev_avg * count + new_value) / (count + 1), count + 1


def save_results(df, csv_path):
    """Save results to CSV."""
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def run_benchmark():
    """Run a single benchmark iteration."""
    print(f"GGNN vs FGC Performance Benchmarking (Mode: {TEST_MODE})")
    print("=" * 80)

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for GGNN benchmarking")
        return

    # Determine CSV file and test parameters
    if TEST_MODE == 'size':
        csv_path = SIZE_CSV
        primary_values = SIZE_RANGE
        secondary_values = SIZE_MODE_DIM_RANGE if ITERATE_DIMS_IN_SIZE_MODE else [FIXED_DIMENSION]
        primary_param = 'size'
        secondary_param = 'dimension'
    else:
        csv_path = DIM_CSV
        primary_values = DIMENSION_RANGE
        secondary_values = DIM_MODE_SIZE_RANGE if ITERATE_SIZES_IN_DIM_MODE else [FIXED_SIZE]
        primary_param = 'dimension'
        secondary_param = 'size'

    # K values to test
    k_values = K_RANGE if ITERATE_OVER_K else [K]

    # Load or create DataFrame
    df = load_or_create_dataframe(csv_path, TEST_MODE)

    total_tests = len(primary_values) * len(secondary_values) * len(k_values)
    print(f"Testing {len(primary_values)} {primary_param} values × {len(secondary_values)} {secondary_param} values × {len(k_values)} K values = {total_tests} total tests")
    print()

    test_count = 0
    for primary_val in primary_values:
        for secondary_val in secondary_values:
            for k_val in k_values:
                test_count += 1
                print(f"Progress: {test_count}/{total_tests} - Testing {primary_param}: {primary_val}, {secondary_param}: {secondary_val}, K: {k_val}")

                # Generate data based on test mode
                if TEST_MODE == 'size':
                    data = generate_data(primary_val, secondary_val)
                    size_val, dim_val = primary_val, secondary_val
                else:
                    data = generate_data(secondary_val, primary_val)
                    size_val, dim_val = secondary_val, primary_val

                # Find existing row or create new one
                if TEST_MODE == 'size':
                    existing_row = df[(df['size'] == size_val) & (df['k'] == k_val)]
                    if ITERATE_DIMS_IN_SIZE_MODE:
                        if 'dimension' not in df.columns:
                            df['dimension'] = FIXED_DIMENSION
                        existing_row = df[(df['size'] == size_val) & (df['dimension'] == dim_val) & (df['k'] == k_val)]
                else:
                    existing_row = df[(df['dimension'] == dim_val) & (df['k'] == k_val)]
                    if ITERATE_SIZES_IN_DIM_MODE:
                        if 'size' not in df.columns:
                            df['size'] = FIXED_SIZE
                        existing_row = df[(df['dimension'] == dim_val) & (df['size'] == size_val) & (df['k'] == k_val)]

                if len(existing_row) == 0:
                    # Create new row
                    if TEST_MODE == 'size':
                        new_row = {'size': size_val, 'k': k_val, 'ggnn_time': 0.0, 'fgc_time': 0.0, 'count': 0}
                        if ITERATE_DIMS_IN_SIZE_MODE:
                            new_row['dimension'] = dim_val
                    else:
                        new_row = {'dimension': dim_val, 'k': k_val, 'ggnn_time': 0.0, 'fgc_time': 0.0, 'count': 0}
                        if ITERATE_SIZES_IN_DIM_MODE:
                            new_row['size'] = size_val
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    row_idx = len(df) - 1
                else:
                    row_idx = existing_row.index[0]

                # Get current values
                current_ggnn_avg = df.loc[row_idx, 'ggnn_time']
                current_fgc_avg = df.loc[row_idx, 'fgc_time']
                current_count = int(df.loc[row_idx, 'count'])

                # Time GGNN
                ggnn_time = time_ggnn(data, k_val)
                if ggnn_time is not None:
                    new_ggnn_avg, new_count_g = update_average(current_ggnn_avg, current_count, ggnn_time)
                    df.loc[row_idx, 'ggnn_time'] = new_ggnn_avg
                    print(f"  GGNN: {ggnn_time:.2f}ms (avg: {new_ggnn_avg:.2f}ms)")
                else:
                    print(f"  GGNN: ERROR")

                # Time FGC
                fgc_time = time_fgc(data, k_val)
                if fgc_time is not None:
                    new_fgc_avg, new_count_f = update_average(current_fgc_avg, current_count, fgc_time)
                    df.loc[row_idx, 'fgc_time'] = new_fgc_avg
                    print(f"  FGC: {fgc_time:.2f}ms (avg: {new_fgc_avg:.2f}ms)")
                else:
                    print(f"  FGC: ERROR")

                # Update count (use whichever was successful)
                if ggnn_time is not None or fgc_time is not None:
                    df.loc[row_idx, 'count'] = current_count + 1

                # Calculate and display speedup
                if ggnn_time is not None and fgc_time is not None:
                    speedup = ggnn_time / fgc_time
                    avg_speedup = new_ggnn_avg / new_fgc_avg if new_fgc_avg > 0 else 0
                    print(f"  Speedup (FGC vs GGNN): {speedup:.2f}x (avg: {avg_speedup:.2f}x)")

                # Save results after each round
                save_results(df, csv_path)
                print()

    print("Benchmarking complete!")
    print(f"Final results saved to {csv_path}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="GGNN vs FGC Performance Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'iterations',
        type=int,
        nargs='?',
        default=1,
        help='Number of times to run the benchmark (default: 1)'
    )

    args = parser.parse_args()

    if args.iterations < 1:
        print("ERROR: Number of iterations must be at least 1")
        return

    print(f"Running benchmark {args.iterations} time(s)")
    print("=" * 80)
    print()

    for i in range(args.iterations):
        if args.iterations > 1:
            print(f"\n{'=' * 80}")
            print(f"ITERATION {i + 1} of {args.iterations}")
            print(f"{'=' * 80}\n")

        run_benchmark()

    print("\n" + "=" * 80)
    print(f"All {args.iterations} iteration(s) complete!")


if __name__ == "__main__":
    main()
