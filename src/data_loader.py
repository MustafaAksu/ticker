# src/data_loader.py
import pandas as pd
from src.config import RAW_DIR, PROC_DIR

def load_bist_data(csv_path="BIST100_OHLCV.csv"):
    df = pd.read_csv(RAW_DIR / csv_path, index_col=0, parse_dates=True)
    close = df.filter(like='_Close').ffill()
    volume = df.filter(like='_Volume').ffill()
    close.columns = [c.replace('_Close', '') for c in close.columns]
    volume.columns = [c.replace('_Volume', '') for c in volume.columns]
    return close, volume