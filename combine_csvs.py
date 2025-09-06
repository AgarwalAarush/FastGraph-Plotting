import pandas as pd
import glob
import os

# Process FAISS CSV files
faiss_files = glob.glob('faiss-csv/*.csv')
faiss_dataframes = []

for file in faiss_files:
    df = pd.read_csv(file)
    faiss_dataframes.append(df)

if faiss_dataframes:
    faiss_combined_df = pd.concat(faiss_dataframes, ignore_index=True)
    faiss_combined_df.to_csv('faiss_data.csv', index=False)
    print(f"Combined {len(faiss_files)} FAISS CSV files into faiss_data.csv")
    print(f"FAISS total rows: {len(faiss_combined_df)}")

# Process ScaNN CSV files
scann_files = glob.glob('scann-csv/*.csv')
scann_dataframes = []

for file in scann_files:
    df = pd.read_csv(file)
    scann_dataframes.append(df)

if scann_dataframes:
    scann_combined_df = pd.concat(scann_dataframes, ignore_index=True)
    scann_combined_df.to_csv('scann_data.csv', index=False)
    print(f"Combined {len(scann_files)} ScaNN CSV files into scann_data.csv")
    print(f"ScaNN total rows: {len(scann_combined_df)}")