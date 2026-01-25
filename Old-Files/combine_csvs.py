import glob
import os
import pandas as pd

ALGORITHMS = [
    ("GGNN", "ggnn-data", "ggnn_data.csv"),
    ("Annoy", "annoy-data", "annoy_data.csv"),
    ("FAISS", "faiss-data", "faiss_data.csv"),
    ("HNSWLIB", "hnswlib-data", "hnswlib_data.csv"),
    ("ScaNN", "scann-data", "scann_data.csv"),
]


def combine_csvs():
    """Combine per-run CSVs into a single CSV per algorithm."""
    for name, directory, output in ALGORITHMS:
        csv_pattern = os.path.join(directory, "*.csv")
        csv_files = sorted(glob.glob(csv_pattern))

        if not csv_files:
            print(f"No {name} CSV files found in {directory}/ directory")
            continue

        dataframes = [pd.read_csv(path) for path in csv_files]
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output, index=False)
        print(f"Combined {len(csv_files)} {name} CSV files into {output}")
        print(f"{name} total rows: {len(combined_df)}")


if __name__ == "__main__":
    combine_csvs()
