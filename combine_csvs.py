import pandas as pd
import glob
import os

process_hnswlib = False
process_annoy = True

# Process HNSWLIB CSV files
hnswlib_files = glob.glob('hnswlib-data/*.csv')
hnswlib_dataframes = []

for file in hnswlib_files:
    df = pd.read_csv(file)
    hnswlib_dataframes.append(df)

if hnswlib_dataframes and process_hnswlib:
    hnswlib_combined_df = pd.concat(hnswlib_dataframes, ignore_index=True)
    hnswlib_combined_df.to_csv('hnswlib_data.csv', index=False)
    print(f"Combined {len(hnswlib_files)} HNSWLIB CSV files into hnswlib_data.csv")
    print(f"HNSWLIB total rows: {len(hnswlib_combined_df)}")
else:
    print("No HNSWLIB CSV files found in hnswlib-data/ directory")

# Process Annoy CSV files
annoy_files = glob.glob('annoy-data/*.csv')
annoy_dataframes = []

for file in annoy_files:
    df = pd.read_csv(file)
    annoy_dataframes.append(df)

if annoy_dataframes and process_annoy:
    annoy_combined_df = pd.concat(annoy_dataframes, ignore_index=True)
    annoy_combined_df.to_csv('annoy_data.csv', index=False)
    print(f"Combined {len(annoy_files)} Annoy CSV files into annoy_data.csv")
    print(f"Annoy total rows: {len(annoy_combined_df)}")
else:
    print("No Annoy CSV files found in annoy-data/ directory")
