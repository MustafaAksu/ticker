# src/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def returns_from_close(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().dropna(how="all")

def resid_by_market_sector(returns: pd.DataFrame,
                           volume: pd.DataFrame,
                           sectors: pd.Series) -> pd.DataFrame:
    """Her günde çapraz-kesit regresyon: ret ~ market_ret + sector dummies."""
    ret = returns.copy()
    
    # Ortak tarihlerde çalış
    common_idx = ret.index.intersection(volume.index)
    if common_idx.empty:
        return pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    
    ret = ret.loc[common_idx]
    vol = volume.loc[common_idx].reindex(columns=ret.columns).fillna(0)

    # Hacim ağırlıklı piyasa getirisi
    vol_sum = vol.sum(axis=1)
    vw_mkt = (ret * vol).sum(axis=1) / (vol_sum + 1e-12)
    vw_mkt = vw_mkt.replace([np.inf, -np.inf], np.nan).fillna(0)

    resid = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

    for dt in ret.index:
        y = ret.loc[dt].dropna()
        if len(y) < 3:  # en az 3 hisse gerekir
            continue

        tickers = y.index
        X = pd.DataFrame({"market": vw_mkt.loc[dt]}, index=tickers)

        # Sektör dummy'leri (NaN → GENEL)
        if len(sectors) > 0:
            sec = sectors.reindex(tickers).fillna("GENEL")
            dummies = pd.get_dummies(sec, prefix="sec", drop_first=True)
            X = X.join(dummies)

        # NaN'leri temizle
        df = X.join(y.rename("ret")).dropna()
        if len(df) < 3:
            continue

        X_clean, y_clean = df.drop("ret", axis=1), df["ret"]
        if X_clean.empty:
            continue

        model = LinearRegression()
        model.fit(X_clean.values, y_clean.values)
        pred = model.predict(X_clean.values)
        resid.loc[dt, X_clean.index] = y_clean.values - pred

    # Kalan NaN → 0 (nötr)
    return resid.fillna(0)

def base_cost_matrix(resid_ret: pd.DataFrame,
                     sectors: pd.Series,
                     close: pd.DataFrame,
                     volume: pd.DataFrame,
                     corr_window: int = 60,
                     alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    """Korelasyon + sektör + likidite cezasından C_base."""
    tickers = resid_ret.columns

    # Korelasyon: NaN'sız, son corr_window
    recent = resid_ret.iloc[-corr_window:]
    R = recent.corr().fillna(0.0)
    R = R.reindex(index=tickers, columns=tickers).fillna(0.0)
    d_corr = np.sqrt(2.0 * (1.0 - R.clip(-1, 1)))
    np.fill_diagonal(d_corr.values, 0.0)

    # Sektör mesafesi
    sector_dist = pd.DataFrame(1.0, index=tickers, columns=tickers)
    if len(sectors) > 0:
        sec_map = sectors.fillna("GENEL")
        for i in tickers:
            for j in tickers:
                if sec_map.get(i, "GENEL") == sec_map.get(j, "GENEL"):
                    sector_dist.loc[i, j] = 0.2
    np.fill_diagonal(sector_dist.values, 0.0)

    # Likidite cezası
    adv = (volume.rolling(60, min_periods=1).mean().iloc[-1] * 
           close.rolling(60, min_periods=1).mean().iloc[-1])
    adv = adv.reindex(tickers).fillna(1e6)
    liq_pen = (1e8 / (adv + 1e6)).values
    liq_mat = np.tile(liq_pen.reshape(1, -1), (len(tickers), 1))

    C = alpha * d_corr.values + beta * sector_dist.values + liq_mat
    C = np.maximum(C, 1e-12)
    return pd.DataFrame(C, index=tickers, columns=tickers)