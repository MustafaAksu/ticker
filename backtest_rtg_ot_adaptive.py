# backtest_rtg_ot_adaptive.py
# Adaptive RTG-OT Backtest with improved phase, directional priors, sector prior, optional UOT,
# full diagnostics, and comprehensive CLI overrides.
#
# Key design:
# - No look-ahead: OT on p_{t-1} -> p_t, trade at t+1
# - Universe by TRAIN ADV (no look-ahead)
# - Residuals: cross-sectional de-mean daily returns
# - Cost C = alpha * corr_distance  + (1-alpha) * liquidity_penalty
#            + directional_leadlag_prior  + sector_prior
# - Optional RTG phase gate: zero-phase Butter bandpass + Hilbert (SciPy if available),
#   fallback to MA-bandpass + FFT-Hilbert
# - Optional Unbalanced OT (UOT) if POT is installed (--uot_rho)
# - Walk-forward monthly tuning on ~120d; regime-aware defaults; quantile threshold on RFS_Z
# - Weekly rebalance (default), turnover & transaction cost
# - Diagnostics: gate_diagnostics.csv, tuning_history.csv, last_params.json, backtest_metrics.json

from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(all="ignore")

# --- Optional SciPy (better filters & Hilbert) ---
_HAVE_SCIPY = False
try:
    from scipy.signal import butter, filtfilt, hilbert as sp_hilbert
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# --- Optional POT (for UOT) ---
_HAVE_POT = False
try:
    import ot
    _HAVE_POT = True
except Exception:
    _HAVE_POT = False

# ----------------- Helpers -----------------
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = int(max(1, w))
    c = np.convolve(x, np.ones(w)/w, mode='same')
    for i in range(w//2):
        c[i] = c[i+1]
        c[-(i+1)] = c[-(i+2)]
    return c

def simple_bandpass(x: np.ndarray, low_p=5, high_p=20) -> np.ndarray:
    # fallback bandpass (period space): short MA - long MA
    return moving_average(x, low_p) - moving_average(x, high_p)

def analytic_signal_fft(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x)
    N = len(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1; h[N//2] = 1; h[1:N//2] = 2
    else:
        h[0] = 1; h[1:(N+1)//2] = 2
    return np.fft.ifft(X * h)

def zero_phase_bandpass_and_hilbert(x: np.ndarray, low_p: int, high_p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Preferred (if SciPy available): zero-phase Butter bandpass + filtfilt + Hilbert.
    Frequency is in cycles/day; periods low_p..high_p -> freqs 1/high_p .. 1/low_p.
    """
    if not _HAVE_SCIPY or len(x) < max(high_p*3, 30):
        # Fallback: MA-bandpass + FFT-Hilbert
        xf = simple_bandpass(x, low_p=low_p, high_p=high_p)
        z  = analytic_signal_fft(xf)
        return np.abs(z), np.angle(z)

    fs = 1.0  # 1 sample/day
    low = 1.0 / max(high_p, 1)
    high = 1.0 / max(low_p, 1)
    nyq = 0.5 * fs
    lowc = min(max(low/nyq, 1e-5), 0.99)
    highc = min(max(high/nyq, lowc + 1e-5), 0.999)

    b, a = butter(3, [lowc, highc], btype='band')
    xf = filtfilt(b, a, x)
    z  = sp_hilbert(xf)
    return np.abs(z), np.angle(z)

def angle_diff(a, b):
    return np.angle(np.exp(1j*(a-b)))

def entropic_sinkhorn_balanced(p, q, C, epsilon=0.05, max_iter=2000, tol=1e-9):
    K = np.exp(-C / max(epsilon, 1e-12))
    u = np.ones_like(p); v = np.ones_like(q); Kp = K @ v
    for _ in range(max_iter):
        u_prev = u
        u = p / np.maximum(Kp, 1e-16)
        Kv = K.T @ u
        v = q / np.maximum(Kv, 1e-16)
        Kp = K @ v
        if np.linalg.norm(u - u_prev, 1) < tol:
            break
    return (u[:, None] * K) * v[None, :]

def entropic_sinkhorn_uot(p, q, C, epsilon=0.05, rho=0.07):
    if not _HAVE_POT:
        # fallback to balanced
        return entropic_sinkhorn_balanced(p, q, C, epsilon=epsilon)
    return ot.unbalanced.sinkhorn_unbalanced(p, q, C, reg=epsilon, reg_m=rho)

def sharpe_ratio(returns: pd.Series, ann=252) -> float:
    if len(returns) < 3: return 0.0
    mu, sd = returns.mean(), returns.std()
    return float((mu / (sd + 1e-12)) * np.sqrt(ann))

def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0: return 0.0
    curve = (1 + series).cumprod()
    return float((curve / curve.cummax() - 1).min())

# ----------------- IO -----------------
def load_ohlcv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    close_cols = [c for c in df.columns if c.endswith("_Close")]
    vol_cols   = [c for c in df.columns if c.endswith("_Volume")]
    assert len(close_cols)>0 and len(vol_cols)>0, "CSV must contain *_Close / *_Volume."
    close  = df[close_cols].ffill(); volume = df[vol_cols].ffill()
    tickers = [c.replace("_Close","") for c in close_cols]
    return close, volume, tickers

def load_sector_map(sector_path: str | None) -> dict[str, str]:
    if not sector_path: return {}
    p = Path(sector_path)
    if not p.exists(): return {}
    if p.suffix.lower() in [".json"]:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            return {k: str(v) for k,v in d.items()}
        except Exception:
            return {}
    else:
        # assume CSV with columns: Ticker, Sector
        try:
            df = pd.read_csv(p)
            if "Ticker" in df.columns and "Sector" in df.columns:
                return dict(zip(df["Ticker"].astype(str), df["Sector"].astype(str)))
        except Exception:
            pass
    return {}

# ------------- Residuals -------------
def residual_returns(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    rets = close.pct_change().dropna(how='all'); resid = rets.copy()
    for dt in rets.index:
        y = rets.loc[dt].astype(float).values; ok = ~np.isnan(y)
        if not np.any(ok): continue
        a = float(np.nanmean(y[ok])); resid.loc[dt] = y - a
    return resid

# ------------- Cost components -------------
def _adv_vector(close: pd.DataFrame, volume: pd.DataFrame, win: int = 60) -> np.ndarray:
    tickers = [c.replace("_Close","") for c in close.columns]
    vroll = volume.rolling(win, min_periods=win//2).mean().iloc[-1]
    proll = close.rolling(win,  min_periods=win//2).mean().iloc[-1]
    adv = []
    for tk in tickers:
        v60 = vroll.get(f"{tk}_Volume", np.nan); p60 = proll.get(f"{tk}_Close",  np.nan)
        v60 = float(v60) if pd.notna(v60) else 0.0; p60 = float(p60) if pd.notna(p60) else 0.0
        adv.append(v60 * p60)
    return np.asarray(adv, dtype=float)

def _leadlag_prior(resid_hist: pd.DataFrame, win: int = 20) -> np.ndarray:
    """
    Directional lead-lag prior: corr( r_i[t-1], r_j[t] ) over last 'win' days.
    Returns an (N,N) matrix S with S_ij >= 0 favoring i->j (reduce cost i->j).
    """
    R = resid_hist.iloc[-(win+1):].copy()
    if R.shape[0] < win+1:
        return np.zeros((R.shape[1], R.shape[1]))
    Rm1 = R.shift(1).dropna()   # t-1
    Rt  = R.loc[Rm1.index]      # t aligned
    try:
        C = np.corrcoef(Rm1.values.T, Rt.values.T)
        N = Rm1.shape[1]
        # top-right block: corr(X, Y) where X=Rm1 (sources), Y=Rt (targets)
        S = C[:N, N:]
        S = np.maximum(S, 0.0)  # positive direction only
        np.fill_diagonal(S, 0.0)
        return S
    except Exception:
        N = R.shape[1]
        return np.zeros((N, N))

def _sector_prior_matrix(tickers_close_cols: list[str], sector_map: dict[str,str], lambda_sector: float) -> np.ndarray:
    """
    Returns (N,N) matrix with -lambda_sector if same sector, 0 otherwise.
    """
    tickers = [c.replace("_Close","") for c in tickers_close_cols]
    sectors = [sector_map.get(tk, None) for tk in tickers]
    N = len(tickers)
    M = np.zeros((N, N), dtype=float)
    if not sector_map: return M
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if sectors[i] is not None and sectors[i] == sectors[j]:
                M[i, j] = -abs(lambda_sector)  # discount cost within sector
    return M

def base_cost_matrix(resid: pd.DataFrame,
                     close: pd.DataFrame,
                     volume: pd.DataFrame,
                     corr_window: int = 60,
                     alpha: float = 0.6,
                     lambda_dir: float = 0.2,
                     leadlag_window: int = 20,
                     sector_map: dict[str,str] | None = None,
                     lambda_sector: float = 0.1) -> pd.DataFrame:
    """
    C_base = alpha * d_corr + (1 - alpha) * liq_mat + C_dir + C_sector
      - d_corr: symmetric correlation distance
      - liq_mat: destination-based liquidity penalty
      - C_dir: directional prior, reduces C_{i->j} when i leads j (asymmetric)
      - C_sector: within-sector discount (optional)
    """
    # Corr distance
    R = resid.iloc[-corr_window:].corr().fillna(0.0).clip(-1, 1)
    d_corr = np.sqrt(2.0 * (1.0 - R)).replace([np.inf,-np.inf], 0.0).fillna(0.0)
    np.fill_diagonal(d_corr.values, 0.0)

    # Liquidity penalty (N,N)
    adv = _adv_vector(close, volume, win=60)
    liq_pen = (1e8 / (adv + 1e6))
    N = d_corr.shape[0]
    liq_mat = np.repeat(liq_pen.reshape(1, -1), N, axis=0)

    # Directional lead-lag prior (N,N)
    S = _leadlag_prior(resid, win=leadlag_window)  # >=0 on favorable directions
    C_dir = -abs(lambda_dir) * S  # reduce cost where leader->follower is strong

    # Sector/community prior (N,N)
    C_sec = _sector_prior_matrix(list(close.columns), sector_map or {}, lambda_sector=lambda_sector)

    # Combine
    C = alpha * d_corr.values + (1 - alpha) * liq_mat + C_dir + C_sec

    # Clamp to positive and sane
    med = float(np.median(C))
    C = np.where(np.isfinite(C), C, med)
    C = np.maximum(C, 1e-12)
    np.fill_diagonal(C, 0.0)
    return pd.DataFrame(C, index=d_corr.index, columns=d_corr.columns)

# ------------- RTG Phase gate -------------
def rtg_phase_gate(signal_df: pd.DataFrame,
                   W=60, low_p=5, high_p=20,
                   w_res=0.5, w_dir=0.3, w_amp=0.2,
                   alpha=1.0, gamma=4.0) -> np.ndarray:
    assert signal_df.shape[0] >= W + high_p + 5, "Not enough data for phase."
    X = signal_df.iloc[-(W + high_p + 5):].copy()
    T, N = X.shape
    A = np.zeros((T,N)); PHI = np.zeros((T,N))
    for j, col in enumerate(X.columns):
        amp, phi = zero_phase_bandpass_and_hilbert(X[col].values, low_p=low_p, high_p=high_p)
        A[:,j], PHI[:,j] = amp, phi
    phi_t = PHI[-1,:]; amp_t = A[-1,:] + 1e-12
    dphi_hist = PHI[-W:,:,None] - PHI[-W:,None,:]
    kappa = np.abs(np.exp(1j*dphi_hist).mean(axis=0))
    dphi = angle_diff(phi_t[:,None], phi_t[None,:])
    r_res = 0.5*(1+np.cos(dphi))
    r_dir = 0.5*(1+np.sin(dphi))
    r_amp = 1.0/(1.0+np.exp(-gamma*(amp_t[:,None]-amp_t[None,:])))
    gate = (np.power(kappa, alpha)) * (w_res*r_res + w_dir*r_dir) * ((1-w_amp)+w_amp*r_amp)
    gate = np.clip(gate, 0.0, 1.0); np.fill_diagonal(gate, 0.0)
    return gate

# ------------- Regime & tuning -------------
@dataclass
class RegimeParams:
    # Cost / OT
    alpha: float
    epsilon: float
    # Directional/sector priors
    lambda_dir: float
    leadlag_window: int
    lambda_sector: float
    # RTG gate
    phase_eta: float
    phase_low: int
    phase_high: int
    phase_W: int
    phase_w_res: float
    phase_w_dir: float
    phase_w_amp: float
    phase_alpha: float
    phase_gamma: float
    # Trading policy
    z_quantile: float
    rebalance_every: int
    # OT type
    uot_rho: float | None

def detect_regime(eqw_rets: pd.Series, lookback_vol: int = 20, hist_win: int = 120) -> str:
    if len(eqw_rets) < lookback_vol + hist_win: return "calm"
    roll_vol = eqw_rets.rolling(lookback_vol).std()
    recent = float(roll_vol.iloc[-1]); hist = roll_vol.iloc[-hist_win:-1].median()
    return "calm" if (recent <= hist) else "turb"

def default_params_for_regime(regime: str) -> RegimeParams:
    if regime == "calm":
        return RegimeParams(
            alpha=0.6, epsilon=0.03, lambda_dir=0.2, leadlag_window=20, lambda_sector=0.10,
            phase_eta=0.7, phase_low=5, phase_high=25, phase_W=60,
            phase_w_res=0.6, phase_w_dir=0.2, phase_w_amp=0.2, phase_alpha=1.0, phase_gamma=4.0,
            z_quantile=0.70, rebalance_every=5, uot_rho=None
        )
    else:
        return RegimeParams(
            alpha=0.5, epsilon=0.05, lambda_dir=0.25, leadlag_window=20, lambda_sector=0.10,
            phase_eta=0.9, phase_low=3, phase_high=30, phase_W=60,
            phase_w_res=0.5, phase_w_dir=0.3, phase_w_amp=0.2, phase_alpha=1.0, phase_gamma=4.0,
            z_quantile=0.80, rebalance_every=5, uot_rho=None
        )

def tiny_grid_for_regime(regime: str):
    if regime == "calm":
        return {"alpha":[0.5,0.6,0.7], "epsilon":[0.03,0.05], "phase_eta":[0.6,0.8], "z_q":[0.65,0.70,0.75]}
    else:
        return {"alpha":[0.5,0.6],     "epsilon":[0.05,0.08], "phase_eta":[0.7,0.9], "z_q":[0.75,0.80,0.85]}

# --- Tuning objective (Sharpe by default) ---
def eval_params_once(close, volume, resid, sector_map, start_idx, end_idx,
                     alpha, epsilon, use_phase, phase_eta, phase_low, phase_high, phase_W,
                     phase_w_res, phase_w_dir, phase_w_amp, phase_alpha, phase_gamma,
                     z_q, k_top, slip_bps, corr_window,
                     lambda_dir, leadlag_window, lambda_sector,
                     uot_rho):
    rets = close.pct_change().dropna(); dates = close.index
    port, dates_bt = [], []; last_w = None
    for t in range(start_idx + corr_window + 21, end_idx):
        # weekly
        if (t - (start_idx + corr_window + 21)) % 5 != 0:
            if last_w is not None and t+1 < len(dates):
                dt1 = dates[t+1]; cols = [c for c in last_w.index if last_w[c]>0]
                r = (last_w[cols] * rets.loc[dt1, cols].fillna(0.0)).sum()
                port.append(float(r)); dates_bt.append(dt1)
            continue
        dtm1, dt0 = dates[t-1], dates[t]
        if resid.index.get_loc(dt0) < corr_window+10: continue
        Cb = base_cost_matrix(resid.loc[:dt0], close.loc[:dt0], volume.loc[:dt0],
                              corr_window=corr_window, alpha=alpha,
                              lambda_dir=lambda_dir, leadlag_window=leadlag_window,
                              sector_map=sector_map, lambda_sector=lambda_sector)
        Cuse = Cb.values.copy()
        if use_phase:
            try:
                gate = rtg_phase_gate(resid.loc[:dt0][Cb.columns],
                                      W=phase_W, low_p=phase_low, high_p=phase_high,
                                      w_res=phase_w_res, w_dir=phase_w_dir, w_amp=phase_w_amp,
                                      alpha=phase_alpha, gamma=phase_gamma)
                Cuse = Cuse * (1 - phase_eta * gate) + 1e-12 * np.median(Cuse)
                np.fill_diagonal(Cuse, 0.0)
            except AssertionError:
                pass

        mcap_tm1 = (close.loc[dtm1]*1e6).reindex(Cb.index); mcap_t = (close.loc[dt0]*1e6).reindex(Cb.index)
        p_tm1 = (mcap_tm1/mcap_tm1.sum()).values; p_t = (mcap_t/mcap_t.sum()).values
        if uot_rho is not None and _HAVE_POT:
            F = entropic_sinkhorn_uot(p_tm1, p_t, Cuse, epsilon=epsilon, rho=uot_rho)
        else:
            F = entropic_sinkhorn_balanced(p_tm1, p_t, Cuse, epsilon=epsilon)

        inflow = F.sum(0) - F.sum(1)
        rfs = inflow / np.maximum(mcap_t.values, 1e-12)
        rfs_z = (rfs - rfs.mean())/(rfs.std(ddof=0)+1e-12)
        s = pd.Series(rfs_z, index=Cb.index).sort_values(ascending=False)
        thr = s.quantile(z_q); s_f = s[s>=thr]
        top_cols = list(s_f.head(k_top).index) if len(s_f)>0 else list(s.head(k_top).index)

        if t+1 < len(dates):
            dt1 = dates[t+1]
            nr = rets.loc[dt1, top_cols].mean()
            w_t = pd.Series(0.0, index=close.columns); w_t.loc[top_cols] = 1/len(top_cols)
            turnover = w_t.abs().sum() if last_w is None else (w_t-last_w).abs().sum()
            port.append(float(nr - slip_bps*float(turnover))); dates_bt.append(dt1); last_w = w_t
    return sharpe_ratio(pd.Series(port, index=pd.Index(dates_bt, name="Date")))

def autotune_params(close, volume, resid, sector_map, t_idx, train_lookback, use_phase, corr_window, overrides: dict):
    start_idx = max(0, t_idx - train_lookback); end_idx = t_idx
    eqw = close.pct_change().mean(axis=1)
    regime = detect_regime(eqw.loc[close.index[start_idx:end_idx]], lookback_vol=20, hist_win=120)
    defaults = default_params_for_regime(regime); grid = tiny_grid_for_regime(regime)

    # apply CLI overrides: fixed if provided
    def fixed_or_list(val, lst): return [val] if val is not None else lst
    grid_alpha = fixed_or_list(overrides.get("cost_alpha"), grid["alpha"])
    grid_eps   = fixed_or_list(overrides.get("epsilon"), grid["epsilon"])
    grid_peta  = fixed_or_list(overrides.get("phase_eta"), grid["phase_eta"])
    grid_zq    = fixed_or_list(overrides.get("z_quantile"), grid["z_q"])

    # non-tuned overrides: directly replace defaults
    for fld, key in [
        ("phase_low","phase_low"), ("phase_high","phase_high"), ("phase_W","phase_W"),
        ("phase_w_res","phase_w_res"), ("phase_w_dir","phase_w_dir"), ("phase_w_amp","phase_w_amp"),
        ("phase_alpha","phase_alpha"), ("phase_gamma","phase_gamma"),
        ("rebalance_every","rebalance_every"),
        ("lambda_dir","lambda_dir"), ("leadlag_window","leadlag_window"),
        ("lambda_sector","lambda_sector"), ("uot_rho","uot_rho")
    ]:
        if overrides.get(key) is not None:
            setattr(defaults, fld, overrides[key])

    best = {"score": -1e9, "alpha": defaults.alpha, "epsilon": defaults.epsilon,
            "phase_eta": defaults.phase_eta, "z_q": defaults.z_quantile}
    for a in grid_alpha:
        for e in grid_eps:
            for pe in grid_peta:
                for zq in grid_zq:
                    sc = eval_params_once(close, volume, resid, sector_map, start_idx, end_idx,
                                          alpha=a, epsilon=e, use_phase=use_phase,
                                          phase_eta=pe, phase_low=defaults.phase_low, phase_high=defaults.phase_high,
                                          phase_W=defaults.phase_W, phase_w_res=defaults.phase_w_res,
                                          phase_w_dir=defaults.phase_w_dir, phase_w_amp=defaults.phase_w_amp,
                                          phase_alpha=defaults.phase_alpha, phase_gamma=defaults.phase_gamma,
                                          z_q=zq, k_top=10, slip_bps=0.001, corr_window=corr_window,
                                          lambda_dir=defaults.lambda_dir, leadlag_window=defaults.leadlag_window,
                                          lambda_sector=defaults.lambda_sector, uot_rho=defaults.uot_rho)
                    if sc > best["score"]:
                        best.update({"score": sc, "alpha": a, "epsilon": e, "phase_eta": pe, "z_q": zq})

    tuned = RegimeParams(
        alpha=best["alpha"], epsilon=best["epsilon"],
        lambda_dir=defaults.lambda_dir, leadlag_window=defaults.leadlag_window,
        lambda_sector=defaults.lambda_sector,
        phase_eta=best["phase_eta"], phase_low=defaults.phase_low, phase_high=defaults.phase_high,
        phase_W=defaults.phase_W, phase_w_res=defaults.phase_w_res, phase_w_dir=defaults.phase_w_dir,
        phase_w_amp=defaults.phase_w_amp, phase_alpha=defaults.phase_alpha, phase_gamma=defaults.phase_gamma,
        z_quantile=best["z_q"], rebalance_every=defaults.rebalance_every, uot_rho=defaults.uot_rho
    )
    return tuned, regime

# ------------- Main backtest -------------
def run_backtest(csv_path: Path,
                 out_dir: Path,
                 k_top: int = 10,
                 use_phase: bool = True,
                 max_tickers: int | None = 20,
                 train_days: int = 252,
                 corr_window: int = 60,
                 slip_bps: float = 0.001,
                 tune_every: int = 20,
                 train_lookback: int = 120,
                 overrides: dict | None = None,
                 sector_path: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    close, volume, _ = load_ohlcv(csv_path)
    sector_map = load_sector_map(sector_path)

    # split & universe (TRAIN ADV)
    need_rows = train_days + corr_window + 30
    if close.shape[0] < need_rows: raise ValueError(f"Not enough rows: {close.shape[0]} < {need_rows}")
    split_idx = train_days
    roll_win = min(60, max(5, split_idx - 5))
    tickers_all = [c.replace("_Close","") for c in close.columns]
    vroll_train = volume.iloc[:split_idx].rolling(roll_win, min_periods=roll_win//2).mean().iloc[-1]
    proll_train = close.iloc[:split_idx].rolling(roll_win,  min_periods=roll_win//2).mean().iloc[-1]
    adv_pairs = []
    for tk in tickers_all:
        v60 = vroll_train.get(f"{tk}_Volume", np.nan); p60 = proll_train.get(f"{tk}_Close",  np.nan)
        v60 = float(v60) if pd.notna(v60) else 0.0; p60 = float(p60) if pd.notna(p60) else 0.0
        adv_pairs.append((tk, v60*p60))
    chosen = list(pd.Series(dict(adv_pairs)).sort_values(ascending=False).head(max_tickers).index) if max_tickers else tickers_all
    close  = close[[f"{t}_Close"  for t in chosen]]; volume = volume[[f"{t}_Volume" for t in chosen]]

    # returns & residuals
    rets  = close.pct_change().dropna()
    resid = residual_returns(close, volume)
    eqw = rets.mean(axis=1)

    # initial params (apply CLI overrides)
    regime = detect_regime(eqw.iloc[:split_idx+1], lookback_vol=20, hist_win=120)
    params = default_params_for_regime(regime)
    for fld, key in [
        ("alpha","cost_alpha"), ("epsilon","epsilon"),
        ("lambda_dir","lambda_dir"), ("leadlag_window","leadlag_window"), ("lambda_sector","lambda_sector"),
        ("phase_eta","phase_eta"), ("phase_low","phase_low"), ("phase_high","phase_high"), ("phase_W","phase_W"),
        ("phase_w_res","phase_w_res"), ("phase_w_dir","phase_w_dir"), ("phase_w_amp","phase_w_amp"),
        ("phase_alpha","phase_alpha"), ("phase_gamma","phase_gamma"),
        ("z_quantile","z_quantile"), ("rebalance_every","rebalance_every"),
        ("uot_rho","uot_rho")
    ]:
        if overrides.get(key) is not None:
            setattr(params, fld, overrides[key])

    # containers
    port_rets, dates_bt, ics = [], [], []
    gate_diag_rows, tune_hist_rows = [], []
    last_w = None
    prev_rfs = None

    # walk-forward
    start_t = split_idx + corr_window + 21
    for t in range(start_t, len(close.index)-1):
        dtm1, dt0, dt1 = close.index[t-1], close.index[t], close.index[t+1]

        # monthly tuning
        if (t - start_t) % tune_every == 0:
            params, regime = autotune_params(close, volume, resid, sector_map, t, train_lookback, use_phase, corr_window, overrides)
            snap = {"Date": str(dt0.date()), "Regime": regime}
            snap.update(asdict(params)); tune_hist_rows.append(snap)
            print(f"[TUNE {dt0.date()}] reg={regime} a={params.alpha:.2f} eps={params.epsilon:.3f} "
                  f"eta={params.phase_eta:.2f} zq={params.z_quantile:.2f} reb={params.rebalance_every} "
                  f"ldir={params.lambda_dir:.2f} UOT={params.uot_rho is not None}")

        do_reb = ((t - start_t) % params.rebalance_every == 0)

        # cost up to t with priors
        if resid.index.get_loc(dt0) < corr_window+10: continue
        Cb = base_cost_matrix(resid.loc[:dt0], close.loc[:dt0], volume.loc[:dt0],
                              corr_window=corr_window, alpha=params.alpha,
                              lambda_dir=params.lambda_dir, leadlag_window=params.leadlag_window,
                              sector_map=sector_map, lambda_sector=params.lambda_sector)
        Cuse = Cb.values.copy(); gate = None
        gate_mean=p90=p99=mod_ratio=np.nan

        if use_phase:
            try:
                gate = rtg_phase_gate(resid.loc[:dt0][Cb.columns],
                                      W=params.phase_W, low_p=params.phase_low, high_p=params.phase_high,
                                      w_res=params.phase_w_res, w_dir=params.phase_w_dir, w_amp=params.phase_w_amp,
                                      alpha=params.phase_alpha, gamma=params.phase_gamma)
                Cuse = Cuse * (1 - params.phase_eta * gate) + 1e-12*np.median(Cuse)
                np.fill_diagonal(Cuse, 0.0)

                # diagnostics
                g = gate[np.isfinite(gate)]
                if g.size>0:
                    gate_mean = float(np.mean(g)); p90 = float(np.quantile(g,0.9)); p99 = float(np.quantile(g,0.99))
                mask = np.ones_like(Cb.values, dtype=bool); np.fill_diagonal(mask, False)
                base = Cb.values; mod = Cuse
                denom = np.maximum(base, 1e-12)
                ratio = ((base - mod)/denom)[mask]
                mod_ratio = float(np.mean(ratio)) if ratio.size>0 else np.nan

                gate_diag_rows.append({
                    "Date": str(dt0.date()),
                    "mean_gate": gate_mean, "p90_gate": p90, "p99_gate": p99,
                    "mean_eta_gate": float(params.phase_eta * (gate_mean if np.isfinite(gate_mean) else 0.0)),
                    "mean_cost_mod_ratio": mod_ratio,
                    "phase_eta": params.phase_eta, "phase_low": params.phase_low, "phase_high": params.phase_high,
                    "phase_W": params.phase_W, "w_res": params.phase_w_res, "w_dir": params.phase_w_dir,
                    "w_amp": params.phase_w_amp, "phase_alpha": params.phase_alpha, "phase_gamma": params.phase_gamma
                })
            except AssertionError:
                pass

        # OT p_{t-1} -> p_t (optionally UOT)
        mcap_tm1 = (close.loc[dtm1]*1e6).reindex(Cb.index); mcap_t = (close.loc[dt0]*1e6).reindex(Cb.index)
        p_tm1 = (mcap_tm1/mcap_tm1.sum()).values; p_t = (mcap_t/mcap_t.sum()).values
        if params.uot_rho is not None and _HAVE_POT:
            F = entropic_sinkhorn_uot(p_tm1, p_t, Cuse, epsilon=params.epsilon, rho=params.uot_rho)
        else:
            F = entropic_sinkhorn_balanced(p_tm1, p_t, Cuse, epsilon=params.epsilon)

        inflow = F.sum(0) - F.sum(1)
        rfs = inflow / np.maximum(mcap_t.values, 1e-12)
        rfs_z = (rfs - rfs.mean())/(rfs.std(ddof=0)+1e-12)

        # mild flow momentum blend
        if prev_rfs is not None:
            rfs_mom = rfs - prev_rfs
            rfs_mom_z = (rfs_mom - rfs_mom.mean())/(rfs_mom.std(ddof=0)+1e-12)
            s = pd.Series(0.5*rfs_z + 0.5*rfs_mom_z, index=Cb.index).sort_values(ascending=False)
        else:
            s = pd.Series(rfs_z, index=Cb.index).sort_values(ascending=False)
        prev_rfs = rfs.copy()

        # quantile threshold
        thr = s.quantile(params.z_quantile); s_f = s[s>=thr]
        top_cols = list(s_f.head(k_top).index) if len(s_f)>0 else list(s.head(k_top).index)

        if do_reb:
            nr = rets.loc[dt1, top_cols].mean()
            w_t = pd.Series(0.0, index=close.columns); w_t.loc[top_cols] = 1/len(top_cols)
            turnover = w_t.abs().sum() if last_w is None else (w_t-last_w).abs().sum()
            port_rets.append(float(nr - slip_bps*float(turnover))); dates_bt.append(dt1); last_w = w_t
        else:
            if last_w is not None:
                cols = [c for c in last_w.index if last_w[c]>0]
                r = (last_w[cols] * rets.loc[dt1, cols].fillna(0.0)).sum()
                port_rets.append(float(r)); dates_bt.append(dt1)

        # IC vs next-day cross-section
        try:
            ic = pd.Series(rfs_z, index=Cb.index).corr(rets.loc[dt1, Cb.index])
            if np.isfinite(ic): ics.append(float(ic))
        except Exception:
            pass

    # outputs
    port = pd.Series(port_rets, index=pd.Index(dates_bt, name='Date'), name='PortRet')
    out_csv = out_dir / "backtest_portfolio_returns.csv"; port.to_csv(out_csv, index=True)
    bench = rets.loc[port.index].mean(axis=1); (out_dir/"benchmark_eqw_returns.csv").write_text(bench.to_csv())

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

    if len(port)>0:
        eq = (1+port).cumprod(); bq = (1+bench).cumprod()
        plt.figure(figsize=(12,4))
        plt.plot(eq.index, eq.values, label="Strategy")
        plt.plot(bq.index, bq.values, label="EqW Benchmark", alpha=0.7)
        plt.title("RTG-OT Backtest (Adaptive + Directional Priors)")
        plt.xlabel("Date"); plt.ylabel("Equity (rebased)")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"equity_curve.png"); plt.close()

    # diagnostics & param logs
    if len(gate_diag_rows)>0:
        pd.DataFrame(gate_diag_rows).to_csv(out_dir/"gate_diagnostics.csv", index=False)
    if len(tune_hist_rows)>0:
        pd.DataFrame(tune_hist_rows).to_csv(out_dir/"tuning_history.csv", index=False)
    with open(out_dir/"last_params.json","w",encoding="utf-8") as f:
        json.dump({"regime": regime, "params": asdict(params)}, f, ensure_ascii=False, indent=2)

    return out_csv, metrics

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/bt_adaptive")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no_phase", action="store_true")
    ap.add_argument("--max_tickers", type=int, default=20)
    ap.add_argument("--train_days", type=int, default=252)
    ap.add_argument("--corr_window", type=int, default=60)
    ap.add_argument("--slip_bps", type=float, default=0.001)
    ap.add_argument("--tune_every", type=int, default=20)
    ap.add_argument("--train_lookback", type=int, default=120)
    ap.add_argument("--sector_map", type=str, default=None, help="Path to sector map (JSON or CSV with Ticker,Sector)")

    # overrides for costs, priors, RTG, OT type
    ap.add_argument("--cost_alpha", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--lambda_dir", type=float, default=None)
    ap.add_argument("--leadlag_window", type=int, default=None)
    ap.add_argument("--lambda_sector", type=float, default=None)
    ap.add_argument("--uot_rho", type=float, default=None, help="Enable Unbalanced OT with rho (requires POT)")

    # RTG gate params
    ap.add_argument("--phase_eta", type=float, default=None)
    ap.add_argument("--phase_low", type=int, default=None)
    ap.add_argument("--phase_high", type=int, default=None)
    ap.add_argument("--phase_W", type=int, default=None)
    ap.add_argument("--phase_w_res", type=float, default=None)
    ap.add_argument("--phase_w_dir", type=float, default=None)
    ap.add_argument("--phase_w_amp", type=float, default=None)
    ap.add_argument("--phase_alpha", type=float, default=None)
    ap.add_argument("--phase_gamma", type=float, default=None)

    # trading policy
    ap.add_argument("--z_quantile", type=float, default=None)
    ap.add_argument("--rebalance_every", type=int, default=None)

    args = ap.parse_args()
    overrides = {
        "cost_alpha": args.cost_alpha, "epsilon": args.epsilon,
        "lambda_dir": args.lambda_dir, "leadlag_window": args.leadlag_window, "lambda_sector": args.lambda_sector,
        "phase_eta": args.phase_eta, "phase_low": args.phase_low, "phase_high": args.phase_high, "phase_W": args.phase_W,
        "phase_w_res": args.phase_w_res, "phase_w_dir": args.phase_w_dir, "phase_w_amp": args.phase_w_amp,
        "phase_alpha": args.phase_alpha, "phase_gamma": args.phase_gamma,
        "z_quantile": args.z_quantile, "rebalance_every": args.rebalance_every,
        "uot_rho": args.uot_rho
    }

    out_csv, metrics = run_backtest(
        csv_path=Path(args.data),
        out_dir=Path(args.out),
        k_top=args.k,
        use_phase=not args.no_phase,
        max_tickers=args.max_tickers,
        train_days=args.train_days,
        corr_window=args.corr_window,
        slip_bps=args.slip_bps,
        tune_every=args.tune_every,
        train_lookback=args.train_lookback,
        overrides=overrides,
        sector_path=args.sector_map
    )
    print("Backtest CSV:", out_csv)
    print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
