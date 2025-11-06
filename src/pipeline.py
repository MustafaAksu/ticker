from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (DEFAULT_DATA_FILE, SECTOR_MAP_FILE, RTGParams, OTParams,
                        SIG_DIR)
from src.data_loader import load_ohlcv, load_sectors
from src.preprocessor import returns_from_close, resid_by_market_sector, base_cost_matrix
from src.rtg_gating import phase_gate
from src.ot_solver import solve_ot
from src.signals import rfs_from_flow, zscore

def run_daily_pipeline(data_csv: Path = DEFAULT_DATA_FILE,
                       date_str: str | None = None,
                       rtg: RTGParams = RTGParams(),
                       otp: OTParams = OTParams(),
                       k_top: int = 10) -> Path:
    close, volume = load_ohlcv(data_csv)
    sectors = load_sectors(SECTOR_MAP_FILE)

    rets = returns_from_close(close)
    resid = resid_by_market_sector(rets, volume, sectors)
    C_base = base_cost_matrix(resid, sectors, close, volume, corr_window=60)

    # RTG kapısı için sinyal: "resid" veya istersen dolar hacmi (volume*close)
    signal_df = resid.loc[close.index.intersection(resid.index)]
    C_phase = phase_gate(C_base, signal_df,
                         W=rtg.W, band_low=rtg.band_low, band_high=rtg.band_high,
                         w_res=rtg.w_res, w_dir=rtg.w_dir, w_amp=rtg.w_amp,
                         alpha=rtg.alpha, gamma=rtg.gamma, eta=rtg.eta)

    # Son iki günün p_t → p_{t+1} dağılımı
    last2 = close.index[-2:]
    mcap_t   = (close.loc[last2[0]] * 1e6).reindex(C_phase.index)  # yaklaşık float
    mcap_tp1 = (close.loc[last2[1]] * 1e6).reindex(C_phase.index)

    p_t   = (mcap_t / mcap_t.sum()).values
    p_tp1 = (mcap_tp1 / mcap_tp1.sum()).values

    F = solve_ot(p_t, p_tp1, C_phase.values, epsilon=otp.epsilon, rho=otp.rho)
    rfs = rfs_from_flow(F, mcap_tp1)
    rfs_z = zscore(rfs)

    today = date_str or datetime.now().strftime("%Y-%m-%d")
    out = SIG_DIR / f"RFS_{today}.csv"
    pd.DataFrame({"RFS": rfs, "RFS_Z": rfs_z}).to_csv(out, index_label="Ticker")
    return out
