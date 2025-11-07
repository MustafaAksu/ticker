# gettickers.py
import yfinance as yf
import pandas as pd

tickers = [
    'ASELS.IS','GARAN.IS','ENKAI.IS','KCHOL.IS','THYAO.IS','TUPRS.IS','ISCTR.IS','FROTO.IS',
    'BIMAS.IS','AKBNK.IS','YKBNK.IS','DSTKF.IS','TCELL.IS','TTKOM.IS','EREGL.IS','SAHOL.IS',
    'SASA.IS','TOASO.IS','SISE.IS','PGSUS.IS','GUBRF.IS','ASTOR.IS','TAVHL.IS','AEFES.IS',
    'KOZAL.IS','EKGYO.IS','PETKM.IS','ULKER.IS','KRDMD.IS','MGROS.IS',
    'AEFES.IS','ALARK.IS','ASTOR.IS','DOHOL.IS','DOAS.IS','DSTKF.IS',
    'EKGYO.IS','GUBRF.IS','KRDMD.IS','KOZAL.IS','MGROS.IS','PGSUS.IS',
    'SASA.IS','TAVHL.IS','TOASO.IS','ULKER.IS'
]

print("BIST100 verisi indiriliyor...")
data = yf.download(tickers, period='5y', interval='1d', auto_adjust=False)

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