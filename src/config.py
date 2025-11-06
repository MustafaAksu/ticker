# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULT_DIR = BASE_DIR / "results"

# Hiperparametreler
RTG_PARAMS = {
    'W': 60,
    'band': (5, 15),
    'w_res': 0.5, 'w_dir': 0.3, 'w_amp': 0.2,
    'alpha': 1.0, 'gamma': 4.0, 'eta': 0.5
}
OT_PARAMS = {'epsilon': 0.05, 'rho': 0.05}  # UOT için