import json
import pandas as pd
from pathlib import Path
from src.config import SECTOR_MAP_FILE

def load_ohlcv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    close_cols  = [c for c in df.columns if c.endswith("_Close")]
    volume_cols = [c for c in df.columns if c.endswith("_Volume")]
    close  = df[close_cols].rename(columns=lambda c: c.replace("_Close", ""))
    volume = df[volume_cols].rename(columns=lambda c: c.replace("_Volume", ""))
    return close.ffill(), volume.ffill()

def load_sectors(json_path: Path | None = None) -> pd.Series:
    p = json_path or SECTOR_MAP_FILE
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        return pd.Series(m)
    # fallback: hepsini "GENEL"
    return pd.Series(dtype="object")
