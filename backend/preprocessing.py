import pandas as pd
import numpy as np
import os

def load_and_merge_data(data_dir=None):
    """
    Loads raw CSV files, merges train, stores, and features datasets,
    extracts calendar features, and cleans missing values.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")
        
    train_path = os.path.join(data_dir, "train.csv")
    stores_path = os.path.join(data_dir, "stores.csv")
    features_path = os.path.join(data_dir, "features.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing '{train_path}'. Please place your downloaded train.csv, stores.csv, and features.csv inside the data/ folder.")
        
    print(f"Loading datasets from {data_dir}...")
    train = pd.read_csv(train_path)
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)
    
    # Merge datasets
    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    
    # Datetime conversion & Feature Engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["IsHolidayInt"] = df["IsHoliday"].astype(int)
    
    # Impute missing MarkDown values with 0
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # Impute missing economic indicators with median
    df["CPI"] = df["CPI"].fillna(df["CPI"].median())
    df["Unemployment"] = df["Unemployment"].fillna(df["Unemployment"].median())
    
    # Sort chronologically
    df = df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)
    
    print(f"Dataset successfully merged! Total records: {len(df)}")
    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    print(df.head())
