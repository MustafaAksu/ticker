import argparse
from pathlib import Path
from src.pipeline import run_daily_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(Path("data/raw/BIST100_OHLCV.csv")))
    ap.add_argument("--date", type=str, default=None)
    ap.add_argument("--balanced", action="store_true",
                    help="Balanced OT (rho=None). Varsayılan: Unbalanced.")
    args = ap.parse_args()

    from src.config import OTParams
    otp = OTParams(epsilon=0.05, rho=None if args.balanced else 0.05)

    path = run_daily_pipeline(Path(args.data), args.date, otp=otp)
    print(f"RFS dosyası üretildi → {path}")
