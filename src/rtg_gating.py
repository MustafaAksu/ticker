import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

def _bandpass(x: np.ndarray, low_p=5, high_p=15, order=3):
    # period→freq
    low, high = 1.0 / high_p, 1.0 / low_p
    nyq = 0.5
    b, a = butter(order, [low/nyq, high/nyq], btype="band", output="ba")
    # sample rate=1 (günlük), filtfilt için yeterli
    return filtfilt(b, a, x, method="gust")

def _angle_diff(a, b):
    return np.angle(np.exp(1j*(a-b)))

def phase_gate(C_base: pd.DataFrame,
               signal_df: pd.DataFrame,
               W: int = 60,
               band_low: int = 5,
               band_high: int = 15,
               w_res: float = 0.5,
               w_dir: float = 0.3,
               w_amp: float = 0.2,
               alpha: float = 1.0,
               gamma: float = 4.0,
               eta: float = 0.5) -> pd.DataFrame:
    """ChatGPT/Grok RTG kapısı ile C_base → C_phase"""
    assert signal_df.shape[0] >= W + band_high + 5, "Faz için yeterli pencere yok."
    X = signal_df.iloc[-(W + band_high + 5):].copy()  # T x N
    T, N = X.shape
    A = np.zeros((T, N))
    PHI = np.zeros((T, N))
    for j, col in enumerate(X.columns):
        xb = _bandpass(X[col].values, band_low, band_high)
        z  = hilbert(xb)
        A[:, j] = np.abs(z)
        PHI[:, j] = np.angle(z)

    phi_t = PHI[-1, :]
    amp_t = A[-1, :] + 1e-12
    dphi_hist = PHI[-W:, :, None] - PHI[-W:, None, :]
    kappa = np.abs(np.exp(1j*dphi_hist).mean(axis=0))  # PLV ∈ [0,1]

    dphi = _angle_diff(phi_t[:, None], phi_t[None, :])
    r_res = 0.5 * (1 + np.cos(dphi))
    r_dir = 0.5 * (1 + np.sin(dphi))
    r_amp = 1.0 / (1.0 + np.exp(-gamma * (amp_t[:, None] - amp_t[None, :])))

    gate = (kappa**alpha) * (w_res*r_res + w_dir*r_dir) * ((1-w_amp) + w_amp*r_amp)
    gate = np.clip(gate, 0.0, 1.0)
    np.fill_diagonal(gate, 0.0)

    Cb = C_base.values
    C_phase = Cb * (1 - eta * gate) + 1e-12 * np.median(Cb)
    np.fill_diagonal(C_phase, 0.0)
    C_phase = np.maximum(C_phase, 1e-12)
    return pd.DataFrame(C_phase, index=C_base.index, columns=C_base.columns)
