import numpy as np
import pandas as pd

def rfs_from_flow(F_ij: np.ndarray, mcap_tp1: pd.Series) -> pd.Series:
    """Net giriş / float_mcap normalize → RFS (z-score değil, ham)."""
    inflow = F_ij.sum(axis=0) - F_ij.sum(axis=1)     # j'ye giriş - j'den çıkış
    rfs = inflow / np.maximum(mcap_tp1.values, 1e-12)
    return pd.Series(rfs, index=mcap_tp1.index)

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)
