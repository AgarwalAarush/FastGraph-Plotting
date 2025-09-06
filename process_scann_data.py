import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('scann_data.csv')

# Merge dimension columns - use fixed_dimension if available, otherwise dimension
df['final_dimension'] = df['fixed_dimension'].fillna(df['dimension'])

# Merge size columns - use fixed_size if available, otherwise use the existing size
df['final_size'] = df['fixed_size'].fillna(df['size'])

# Create a clean dataframe with merged columns
clean_df = pd.DataFrame({
    'size': df['final_size'],
    'dimension': df['final_dimension'],
    'k': df['k'],
    'scann_time': df['scann_time'],
    'fgc_time': df['fgc_time'],
    'count': df['count']
})

# Group by size, dimension, and k, then calculate weighted averages
grouping_cols = ['size', 'dimension', 'k']

def weighted_avg(group):
    weights = group['count']
    return pd.Series({
        'scann_time': np.average(group['scann_time'], weights=weights),
        'fgc_time': np.average(group['fgc_time'], weights=weights),
        'count': group['count'].sum()
    })

# Group and aggregate
result = clean_df.groupby(grouping_cols).apply(weighted_avg).reset_index()

# Save to new CSV file
result.to_csv('scann_data_impr.csv', index=False)
print(f"Processed {len(df)} rows into {len(result)} rows")
print(f"Saved to scann_data_impr.csv")