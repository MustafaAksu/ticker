# run_pipeline.py
from src.pipeline import run_daily_pipeline
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
run_daily_pipeline(today)
print(f"RFS güncellendi: results/signals/RFS_{today}.csv")