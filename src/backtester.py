import numpy as np
import pandas as pd

def long_only_topk(signal_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Her gün en yüksek k skoru eşit ağırlıkla long."""
    ranks = signal_df.rank(axis=1, ascending=False, method="first")
    mask  = (ranks <= k).astype(float)
    w = mask.div(mask.sum(axis=1), axis=0).fillna(0.0)
    return w

def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    return (weights.shift(1).fillna(0.0) * returns.loc[weights.index]).sum(axis=1)

def metrics(port: pd.Series) -> dict:
    ann = 252
    mu, sd = port.mean(), port.std()
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(ann)
    cum = (1 + port).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    hit = (port > 0).mean()
    return {"Sharpe": float(sharpe), "HitRate": float(hit), "MaxDD": float(mdd),
            "TotalRet": float(cum.iloc[-1] - 1)}
