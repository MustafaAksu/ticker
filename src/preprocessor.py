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
    vw_mkt = (ret * volume.loc[ret.index]).sum(1) / volume.loc[ret.index].sum(1)
    for dt in ret.index:
        y = ret.loc[dt].dropna()
        tickers = y.index
        X = pd.DataFrame({"market": vw_mkt.loc[dt]}, index=tickers)
        if len(sectors) > 0:
            X["sector"] = sectors.reindex(tickers).fillna("GENEL")
            X = pd.get_dummies(X, columns=["sector"], drop_first=True)
        model = LinearRegression()
        model.fit(X.values, y.values)
        y_hat = model.predict(X.values)
        ret.loc[dt, tickers] = y.values - y_hat
    return ret

def base_cost_matrix(resid_ret: pd.DataFrame,
                     sectors: pd.Series,
                     close: pd.DataFrame,
                     volume: pd.DataFrame,
                     corr_window: int = 60,
                     alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    """Korelasyon + sektör + likidite cezasından C_base."""
    tickers = resid_ret.columns
    R = resid_ret.iloc[-corr_window:].corr().fillna(0.0)
    d_corr = np.sqrt(2.0 * (1.0 - R.clip(-1, 1)))
    np.fill_diagonal(d_corr.values, 0.0)

    # aynı sektöre düşük mesafe
    sector_dist = pd.DataFrame(1.0, index=tickers, columns=tickers)
    if len(sectors) > 0:
        for i in tickers:
            for j in tickers:
                if sectors.get(i, "GENEL") == sectors.get(j, "GENEL"):
                    sector_dist.loc[i, j] = 0.2
    np.fill_diagonal(sector_dist.values, 0.0)

    # likidite cezası: hedef j'nin ADV'sine bağlı (TL ~ fiyat*vol)
    adv = (volume.rolling(60).mean().iloc[-1] * close.rolling(60).mean().iloc[-1]).reindex(tickers)
    liq_pen = (1e8 / (adv + 1e6)).values  # küçük ADV → büyük ceza
    liq_mat = np.tile(liq_pen.reshape(1, -1), (len(tickers), 1))

    C = alpha * d_corr.values + beta * sector_dist.values + liq_mat
    C = np.maximum(C, 1e-12)
    return pd.DataFrame(C, index=tickers, columns=tickers)
