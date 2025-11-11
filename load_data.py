# load_data.py
import pandas as pd

def load_data():
    # Load Walmart dataset
    df = pd.read_csv("walmart_10000.csv")

    # Rename columns for safety
    df.columns = [col.strip().lower() for col in df.columns]

    # Assume the dataset has 'Date' and 'Sales' columns
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    print("✅ Data Loaded Successfully!")
    print(df.head())
    return df
