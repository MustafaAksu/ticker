# backtest_rtg_ot.py
# RTG-OT (phase-gated optional) · Entropic OT · Long-only Top-k Backtest
# - Universe picked by TRAIN ADV (no look-ahead)
# - Robust residual_returns (de-mean only; SVD-free)
# - C_base: corr distance + ADV liquidity penalty (shape-safe)
# - Optional RTG phase gate (bandpass + FFT-Hilbert)
# - Daily Top-k long, next-day realization
# - Turnover + TC, Benchmark, IC
# - LOOK-AHEAD SAFE: OT uses p_{t-1} -> p_t and we trade at t+1

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== Small global safety ==============
np.seterr(all="ignore")  # silence harmless warnings in corr etc.

# -----------------------------
# Helpers
# -----------------------------
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = int(max(1, w))
    c = np.convolve(x, np.ones(w)/w, mode='same')
    for i in range(w//2):
        c[i] = c[i+1]
        c[-(i+1)] = c[-(i+2)]
    return c

def simple_bandpass(x: np.ndarray, low_p=5, high_p=20) -> np.ndarray:
    return moving_average(x, low_p) - moving_average(x, high_p)

def analytic_signal_fft(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x)
    N = len(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1; h[N//2] = 1; h[1:N//2] = 2
    else:
        h[0] = 1; h[1:(N+1)//2] = 2
    z = np.fft.ifft(X * h)
    return z

def angle_diff(a, b):
    return np.angle(np.exp(1j*(a-b)))

def entropic_sinkhorn(p, q, C, epsilon=0.05, max_iter=2000, tol=1e-9):
    K = np.exp(-C / max(epsilon, 1e-12))
    u = np.ones_like(p)
    v = np.ones_like(q)
    Kp = K @ v
    for _ in range(max_iter):
        u_prev = u
        u = p / np.maximum(Kp, 1e-16)
        Kv = K.T @ u
        v = q / np.maximum(Kv, 1e-16)
        Kp = K @ v
        if np.linalg.norm(u - u_prev, 1) < tol:
            break
    return (u[:, None] * K) * v[None, :]

def sharpe_ratio(returns: pd.Series, ann=252) -> float:
    mu, sd = returns.mean(), returns.std()
    return float((mu / (sd + 1e-12)) * np.sqrt(ann))

def max_drawdown(series: pd.Series) -> float:
    curve = (1 + series).cumprod()
    return float((curve / curve.cummax() - 1).min())

# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_ohlcv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    close_cols = [c for c in df.columns if c.endswith("_Close")]
    vol_cols   = [c for c in df.columns if c.endswith("_Volume")]
    assert len(close_cols) > 0 and len(vol_cols) > 0, "CSV must contain *_Close / *_Volume columns."
    close  = df[close_cols].ffill()
    volume = df[vol_cols].ffill()
    tickers = [c.replace("_Close","") for c in close_cols]
    return close, volume, tickers

def residual_returns(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional de-mean each day (robust)."""
    rets = close.pct_change().dropna(how='all')
    resid = rets.copy()
    for dt in rets.index:
        y = rets.loc[dt].astype(float).values
        ok = ~np.isnan(y)
        if not np.any(ok):
            continue
        a = float(np.nanmean(y[ok]))
        y_hat = np.full_like(y, a)
        resid.loc[dt] = y - y_hat
    return resid

# -----------------------------
# Cost matrix (HARDENED against 20x400)
# -----------------------------
def _adv_vector(close: pd.DataFrame, volume: pd.DataFrame, win: int = 60) -> np.ndarray:
    """
    Returns an (N,) numpy array of ADV per ticker,
    computed explicitly by looping tickers to avoid Pandas outer-product pitfalls.
    """
    tickers = [c.replace("_Close", "") for c in close.columns]
    adv_vals = []
    vroll = volume.rolling(win, min_periods=win//2).mean().iloc[-1]
    proll = close.rolling(win,  min_periods=win//2).mean().iloc[-1]
    for tk in tickers:
        v60 = vroll.get(f"{tk}_Volume", np.nan)
        p60 = proll.get(f"{tk}_Close",  np.nan)
        v60 = float(v60) if pd.notna(v60) else 0.0
        p60 = float(p60) if pd.notna(p60) else 0.0
        adv_vals.append(v60 * p60)
    return np.asarray(adv_vals, dtype=float)

def base_cost_matrix(resid: pd.DataFrame,
                     close: pd.DataFrame,
                     volume: pd.DataFrame,
                     corr_window: int = 60,
                     alpha: float = 0.7) -> pd.DataFrame:
    """
    C_base = alpha * d_corr + (1 - alpha) * liquidity_penalty
    liquidity_penalty: target j's ADV (60d mean volume * price).
    Shape-safe: never uses Series*Series broadcasting.
    """
    # Corr distance
    R = resid.iloc[-corr_window:].corr()
    R = R.fillna(0.0).clip(-1, 1)
    d_corr = np.sqrt(2.0 * (1.0 - R))
    d_corr = d_corr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    np.fill_diagonal(d_corr.values, 0.0)

    # ADV vector (N,)
    adv = _adv_vector(close, volume, win=60)   # shape: (N,)
    liq_pen = (1e8 / (adv + 1e6))              # (N,)

    N = d_corr.shape[0]
    liq_mat = np.repeat(liq_pen.reshape(1, -1), N, axis=0)  # (N,N)

    C = alpha * d_corr.values + (1 - alpha) * liq_mat

    # Guard: if someone reintroduced an outer product elsewhere
    if C.shape != (N, N):
        # Rebuild strictly via numpy
        liq_mat = np.repeat(liq_pen.reshape(1, -1), N, axis=0)
        C = alpha * d_corr.values + (1 - alpha) * liq_mat

    C = np.maximum(C, 1e-12)
    np.fill_diagonal(C, 0.0)
    assert C.shape == (N, N), f"Shape mismatch: C={C.shape}, expected {(N,N)}"
    return pd.DataFrame(C, index=d_corr.index, columns=d_corr.columns)

# -----------------------------
# Phase gate (optional)
# -----------------------------
def rtg_phase_gate(signal_df: pd.DataFrame,
                   W=60, low_p=5, high_p=20,
                   w_res=0.5, w_dir=0.3, w_amp=0.2,
                   alpha=1.0, gamma=4.0) -> np.ndarray:
    assert signal_df.shape[0] >= W + high_p + 5, "Not enough data for phase."
    X = signal_df.iloc[-(W + high_p + 5):].copy()  # T x N
    T, N = X.shape
    A = np.zeros((T, N)); PHI = np.zeros((T, N))
    for j, col in enumerate(X.columns):
        xb = simple_bandpass(X[col].values, low_p=low_p, high_p=high_p)
        z  = analytic_signal_fft(xb)
        A[:, j] = np.abs(z); PHI[:, j] = np.angle(z)
    phi_t = PHI[-1, :]; amp_t = A[-1, :] + 1e-12
    dphi_hist = PHI[-W:, :, None] - PHI[-W:, None, :]
    kappa = np.abs(np.exp(1j*dphi_hist).mean(axis=0))
    dphi = angle_diff(phi_t[:, None], phi_t[None, :])
    r_res = 0.5 * (1 + np.cos(dphi))
    r_dir = 0.5 * (1 + np.sin(dphi))
    r_amp = 1.0 / (1.0 + np.exp(-gamma * (amp_t[:, None] - amp_t[None, :])))
    gate = (kappa**alpha) * (w_res*r_res + w_dir*r_dir) * ((1-w_amp) + w_amp*r_amp)
    gate = np.clip(gate, 0.0, 1.0); np.fill_diagonal(gate, 0.0)
    return gate

# -----------------------------
# Backtest
# -----------------------------
def run_backtest(csv_path: Path,
                 out_dir: Path,
                 k_top: int = 10,
                 epsilon: float = 0.05,
                 use_phase: bool = True,
                 phase_eta: float = 0.5,
                 max_tickers: int | None = 30,
                 train_days: int = 252,
                 corr_window: int = 60,
                 slip_bps: float = 0.001,   # 10 bps
                 phase_low: int = 5,
                 phase_high: int = 20,
                 phase_W: int = 60) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    close, volume, _ = load_ohlcv(csv_path)

    # --- Train/Test split
    if close.shape[0] < train_days + corr_window + 6:
        raise ValueError(f"Not enough rows: {close.shape[0]}. Need ~{train_days + corr_window + 6}.")
    split_idx = train_days
    train_idx = close.index[:split_idx]
    test_idx  = close.index[split_idx:]

    # --- Universe by TRAIN ADV (no look-ahead)
    if max_tickers is not None:
        roll_win = min(60, max(5, split_idx - 5))
        tickers_all = [c.replace("_Close", "") for c in close.columns]
        adv_pairs = []
        vroll_train = volume.iloc[:split_idx].rolling(roll_win, min_periods=roll_win//2).mean().iloc[-1]
        proll_train = close.iloc[:split_idx].rolling(roll_win,  min_periods=roll_win//2).mean().iloc[-1]
        for tk in tickers_all:
            v60 = vroll_train.get(f"{tk}_Volume", np.nan)
            p60 = proll_train.get(f"{tk}_Close",  np.nan)
            v60 = float(v60) if pd.notna(v60) else 0.0
            p60 = float(p60) if pd.notna(p60) else 0.0
            adv_pairs.append((tk, v60 * p60))
        adv_train = pd.Series(dict(adv_pairs)).sort_values(ascending=False)
        chosen = list(adv_train.head(max_tickers).index)  # bare tickers

        close  = close[[f"{t}_Close"  for t in chosen]]
        volume = volume[[f"{t}_Volume" for t in chosen]]

    # --- Returns & residuals
    rets  = close.pct_change().dropna()
    resid = residual_returns(close, volume)

    # Containers
    port_rets, dates_bt = [], []
    ics = []
    last_w = None

    # Iterate (LOOK-AHEAD SAFE): use p_{t-1} -> p_t, trade at t+1
    for t in range(split_idx + corr_window + 21, len(close.index) - 1):
        dtm1 = close.index[t-1]  # t-1
        dt0  = close.index[t]    # t
        dt1  = close.index[t+1]  # t+1

        resid_hist = resid.loc[:dt0]
        if resid_hist.shape[0] < corr_window + 10:
            continue

        C_base = base_cost_matrix(resid_hist, close.loc[:dt0], volume.loc[:dt0],
                                  corr_window=corr_window, alpha=0.7)

        C_use = C_base.values.copy()
        if use_phase:
            try:
                gate = rtg_phase_gate(resid_hist[C_base.columns],
                                      W=phase_W, low_p=phase_low, high_p=phase_high,
                                      w_res=0.5, w_dir=0.3, w_amp=0.2, alpha=1.0, gamma=4.0)
                C_use = C_use * (1 - phase_eta * gate) + 1e-12 * np.median(C_use)
                np.fill_diagonal(C_use, 0.0)
            except AssertionError:
                pass

        # OT on last completed change
        mcap_tm1 = (close.loc[dtm1] * 1e6).reindex(C_base.index)
        mcap_t   = (close.loc[dt0]  * 1e6).reindex(C_base.index)
        p_tm1 = (mcap_tm1 / mcap_tm1.sum()).values
        p_t   = (mcap_t   / mcap_t.sum()).values

        F = entropic_sinkhorn(p_tm1, p_t, C_use, epsilon=epsilon, max_iter=2000, tol=1e-9)

        inflow = F.sum(axis=0) - F.sum(axis=1)
        rfs    = inflow / np.maximum(mcap_t.values, 1e-12)
        rfs_z  = (rfs - rfs.mean()) / (rfs.std(ddof=0) + 1e-12)
        s      = pd.Series(rfs_z, index=C_base.index).sort_values(ascending=False)

        top_cols = list(s.head(k_top).index)

        # Realize at t+1
        try:
            next_rets = rets.loc[dt1, top_cols].values
        except KeyError:
            norm = [c if c.endswith("_Close") else f"{c}_Close" for c in top_cols]
            next_rets = rets.loc[dt1, norm].values
        port_ret = float(np.nanmean(next_rets))

        # Turnover + TC
        w_t = pd.Series(0.0, index=close.columns)
        w_t.loc[top_cols] = 1.0 / len(top_cols)
        turnover = w_t.abs().sum() if last_w is None else (w_t - last_w).abs().sum()
        tc = slip_bps * float(turnover)

        port_rets.append(port_ret - tc); dates_bt.append(dt1)
        last_w = w_t

        # IC: today's signal vs tomorrow's cross-section
        try:
            ic = pd.Series(rfs_z, index=C_base.index).corr(rets.loc[dt1, C_base.index])
            if np.isfinite(ic):
                ics.append(float(ic))
        except Exception:
            pass

    port = pd.Series(port_rets, index=pd.Index(dates_bt, name='Date'), name='PortRet')
    out_csv = out_dir / "backtest_portfolio_returns.csv"
    port.to_csv(out_csv, index=True)

    # Benchmark (rough EqW in universe)
    bench = rets.loc[port.index].mean(axis=1)
    bench.to_csv(out_dir / "benchmark_eqw_returns.csv", index=True)

    metrics = {
        "Start": str(port.index.min().date()) if len(port)>0 else None,
        "End":   str(port.index.max().date()) if len(port)>0 else None,
        "Ndays": int(port.shape[0]),
        "Sharpe": sharpe_ratio(port) if len(port)>5 else None,
        "HitRate": float((port>0).mean()) if len(port)>0 else None,
        "MaxDD": max_drawdown(port) if len(port)>5 else None,
        "TotalReturn": float((1+port).prod()-1) if len(port)>0 else None,
        "Bench_Sharpe": sharpe_ratio(bench) if len(bench)>5 else None,
        "Bench_TotalReturn": float((1+bench).prod()-1) if len(bench)>0 else None,
        "IC_mean": float(np.mean(ics)) if len(ics)>0 else None,
        "IC_stderr": float(np.std(ics, ddof=1)/np.sqrt(len(ics))) if len(ics)>1 else None,
        "IC_N": int(len(ics))
    }
    with open(out_dir/"backtest_metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Equity curves
    if len(port)>0:
        eq = (1+port).cumprod()
        bq = (1+bench).cumprod()
        plt.figure(figsize=(12,4))
        plt.plot(eq.index, eq.values, label="Strategy")
        plt.plot(bq.index, bq.values, label="EqW Benchmark", alpha=0.7)
        plt.title("RTG-OT Backtest Equity (Long-only Top-k)")
        plt.xlabel("Date"); plt.ylabel("Equity (rebased)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir/"equity_curve.png")
        plt.close()

    return out_csv, metrics

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to OHLCV CSV (Date, TICKER_Close, TICKER_Volume...)")
    ap.add_argument("--out", type=str, default="results/backtest_demo", help="Output directory")
    ap.add_argument("--k", type=int, default=10, help="Top-k longs each day")
    ap.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn entropic reg")
    ap.add_argument("--no_phase", action="store_true", help="Disable RTG phase gating")
    ap.add_argument("--phase_eta", type=float, default=0.5, help="Phase gate strength (0..1)")
    ap.add_argument("--max_tickers", type=int, default=30, help="Universe size (picked by TRAIN ADV)")
    ap.add_argument("--train_days", type=int, default=252, help="Train window length (business days)")
    ap.add_argument("--corr_window", type=int, default=60, help="Correlation window for d_corr")
    ap.add_argument("--slip_bps", type=float, default=0.001, help="Transaction cost per unit turnover (e.g., 0.001=10bps)")
    ap.add_argument("--phase_low", type=int, default=5)
    ap.add_argument("--phase_high", type=int, default=20)
    ap.add_argument("--phase_W", type=int, default=60)
    args = ap.parse_args()

    out_csv, metrics = run_backtest(
        csv_path=Path(args.data),
        out_dir=Path(args.out),
        k_top=args.k,
        epsilon=args.epsilon,
        use_phase=not args.no_phase,
        phase_eta=args.phase_eta,
        max_tickers=args.max_tickers,
        train_days=args.train_days,
        corr_window=args.corr_window,
        slip_bps=args.slip_bps,
        phase_low=args.phase_low,
        phase_high=args.phase_high,
        phase_W=args.phase_W,
    )
    print("Backtest CSV:", out_csv)
    print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
