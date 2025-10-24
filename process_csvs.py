
import pandas as pd
import numpy as np
import os

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

    # Ensure all necessary columns are present
    required_cols = key_cols + time_cols + ['count']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns in {file_path}. Skipping.")
        return

    # Define aggregation logic
    agg_funs = {}
    for col in time_cols:
        agg_funs[col] = lambda x: np.average(x, weights=df.loc[x.index, 'count'])

    agg_funs['count'] = 'sum'

    # Group and aggregate
    df_cleaned = df.groupby(key_cols).agg(agg_funs).reset_index()

    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


def main():
    """Main function to process all algorithm CSVs."""
    
    algorithms_to_process = [
        {
            'name': 'FAISS',
            'file': 'faiss_data.csv',
            'time_cols': ['faiss_time', 'fgc_time'],
        },
        {
            'name': 'ScaNN',
            'file': 'scann_data.csv',
            'time_cols': ['scann_time', 'fgc_time'],
        },
        {
            'name': 'HNSWLIB',
            'file': 'hnswlib_data.csv',
            'time_cols': ['hnswlib_time', 'fgc_time'],
        },
        {
            'name': 'Annoy',
            'file': 'annoy_data.csv',
            'time_cols': ['annoy_time', 'fgc_time'],
        }
    ]

    key_columns = ['size', 'k', 'dimension']
    
    # In scann_data.csv, the dimension column is named 'dimension' not 'dims'
    # so no special handling is needed if the file is consistent.
    # We will add a check for 'dims' and rename it for compatibility.
    
    print("Starting CSV processing...")
    
    for algo in algorithms_to_process:
        file_path = algo['file']
        df = pd.read_csv(file_path)
        if 'dims' in df.columns and 'dimension' not in df.columns:
            df = df.rename(columns={'dims': 'dimension'})
            df.to_csv(file_path, index=False) # save it back for consistency
            print(f"Renamed 'dims' to 'dimension' in {file_path}")

        output_file = file_path.replace('.csv', '_cleaned.csv')
        process_csv(file_path, output_file, algo['time_cols'], key_columns)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
