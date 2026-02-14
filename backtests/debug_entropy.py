"""Debug: Check SampEn values and z-score distribution."""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from entropy_signal import calculate_indicators as entropy_indicators

symbols = ['EURUSD=X', 'BTC-USD', 'LINK-USD']
end = datetime.now()
start = end - timedelta(days=2*365+30)

for sym in symbols:
    print(f"\n=== {sym} ===")
    df = yf.Ticker(sym).history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d')
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    df_ind = entropy_indicators(df)
    
    valid = df_ind['sampen_zscore'].dropna()
    print(f"  SampEn valid: {df_ind['sampen'].notna().sum()}/{len(df_ind)}")
    print(f"  Z-score valid: {len(valid)}/{len(df_ind)}")
    if len(valid) > 0:
        print(f"  Z-score stats: min={valid.min():.3f}, max={valid.max():.3f}, mean={valid.mean():.3f}, std={valid.std():.3f}")
        print(f"  Z < -1.5: {(valid < -1.5).sum()} bars")
        print(f"  Z > +1.5: {(valid > 1.5).sum()} bars")
        print(f"  Z < -1.0: {(valid < -1.0).sum()} bars")
        print(f"  Z > +1.0: {(valid > 1.0).sum()} bars")
        
    # Also check raw SampEn distribution
    se_valid = df_ind['sampen'].dropna()
    if len(se_valid) > 0:
        print(f"  SampEn stats: min={se_valid.min():.4f}, max={se_valid.max():.4f}, mean={se_valid.mean():.4f}")
        inf_count = np.isinf(se_valid).sum()
        nan_count = se_valid.isna().sum()
        print(f"  SampEn inf: {inf_count}, nan: {nan_count}")
