# gettickers.py
import yfinance as yf
import pandas as pd

tickers = [
    'THYAO.IS','GARAN.IS','AKBNK.IS','ISCTR.IS','YKBNK.IS',
    'ASELS.IS','EREGL.IS','SISE.IS','PETKM.IS','TUPRS.IS',
    'SAHOL.IS','KCHOL.IS','BIMAS.IS','TTKOM.IS','ARCLK.IS',
    'FROTO.IS','VESTL.IS','TCELL.IS','HALKB.IS','ENKAI.IS'
]

print("BIST100 verisi indiriliyor...")
data = yf.download(tickers, period='2y', interval='1d', auto_adjust=False)

# Close ve Volume'u düzgün al
close = data['Close']      # (tarih, ticker)
volume = data['Volume']    # (tarih, ticker)

# CSV formatına çevir: THYAO_Close, THYAO_Volume...
df = pd.DataFrame()
for t in tickers:
    short = t.replace('.IS', '')
    df[f'{short}_Close'] = close[t]
    df[f'{short}_Volume'] = volume[t]

# Kaydet
df.to_csv('data/raw/BIST100_OHLCV.csv')
print(f"Veri kaydedildi: data/raw/BIST100_OHLCV.csv ({len(df)} gün)")