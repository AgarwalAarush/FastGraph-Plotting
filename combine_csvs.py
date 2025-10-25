import pandas as pd
import glob
import os

process_ggnn = True
process_annoy = False

# Process GGNN CSV files
ggnn_files = glob.glob('ggnn-data/*.csv')
ggnn_dataframes = []

for file in ggnn_files:
    df = pd.read_csv(file)
    ggnn_dataframes.append(df)

if ggnn_dataframes and process_ggnn:
    ggnn_combined_df = pd.concat(ggnn_dataframes, ignore_index=True)
    ggnn_combined_df.to_csv('ggnn_data.csv', index=False)
    print(f"Combined {len(ggnn_files)} GGNN CSV files into ggnn_data.csv")
    print(f"GGNN total rows: {len(ggnn_combined_df)}")
else:
    print("No GGNN CSV files found in ggnn-data/ directory")

# # Process Annoy CSV files
# annoy_files = glob.glob('annoy-data/*.csv')
# annoy_dataframes = []

# for file in annoy_files:
#     df = pd.read_csv(file)
#     annoy_dataframes.append(df)

# if annoy_dataframes and process_annoy:
#     annoy_combined_df = pd.concat(annoy_dataframes, ignore_index=True)
#     annoy_combined_df.to_csv('annoy_data.csv', index=False)
#     print(f"Combined {len(annoy_files)} Annoy CSV files into annoy_data.csv")
#     print(f"Annoy total rows: {len(annoy_combined_df)}")
# else:
#     print("No Annoy CSV files found in annoy-data/ directory")
