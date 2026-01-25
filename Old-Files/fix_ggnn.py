import pandas as pd
import numpy as np

def fix_ggnn_data():
    """
    Fixes missing size values in ggnn_data.csv by setting them to 1M (1000000).
    """
    file_path = 'ggnn_data.csv'
    
    try:
        # Load the CSV file
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Check initial state
        print(f"Total rows: {len(df)}")
        
        # Find rows with missing or invalid size values
        missing_size_mask = df['size'].isna() | (df['size'] == '') | (df['size'] == 0)
        missing_count = missing_size_mask.sum()
        
        print(f"Rows with missing/invalid size: {missing_count}")
        
        if missing_count > 0:
            # Update missing size values to 1M (1000000)
            df.loc[missing_size_mask, 'size'] = 1000000
            print(f"Updated {missing_count} rows to size = 1M (1000000)")
            
            # Save back to the same file
            df.to_csv(file_path, index=False)
            print(f"Saved updated data back to {file_path}")
        else:
            print("No missing size values found. No changes needed.")
            
        # Show summary of size values
        print("\nSize value counts:")
        print(df['size'].value_counts().sort_index())
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please make sure the file exists.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    fix_ggnn_data()
