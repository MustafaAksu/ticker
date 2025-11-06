from dataclasses import dataclass
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
META_DIR   = DATA_DIR / "metadata"
RESULT_DIR = BASE_DIR / "results"
FIG_DIR    = RESULT_DIR / "figures"
SIG_DIR    = RESULT_DIR / "signals"
BT_DIR     = RESULT_DIR / "backtests"

for d in [RAW_DIR, PROC_DIR, META_DIR, RESULT_DIR, FIG_DIR, SIG_DIR, BT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class RTGParams:
    W: int = 60
    band_low: int = 5
    band_high: int = 15
    w_res: float = 0.5
    w_dir: float = 0.3
    w_amp: float = 0.2
    alpha: float = 1.0
    gamma: float = 4.0
    eta: float = 0.5

@dataclass
class OTParams:
    epsilon: float = 0.05     # entropik reg.
    rho: float | None = 0.05  # unbalanced OT; None ise balanced

DEFAULT_DATA_FILE = RAW_DIR / "BIST100_OHLCV.csv"
SECTOR_MAP_FILE   = META_DIR / "sector_map.json"
