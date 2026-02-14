#!/usr/bin/env python3
"""
Analyze forex vs stocks volatility and adjust parameters
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from engine import fetch_data

def analyze_volatility(symbol, interval='1h', days=30):
    """Calculate volatility metrics"""
    df = fetch_data(symbol, interval, days)
    if len(df) < 20:
        return None
    
    # ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # ATR as % of price
    atr_pct = (atr / df['Close'].iloc[-1]) * 100
    
    # Daily range %
    daily_range_pct = ((df['High'] - df['Low']) / df['Close']).mean() * 100
    
    # Directional movement
    returns = df['Close'].pct_change().dropna()
    up_moves = returns[returns > 0].sum()
    down_moves = returns[returns < 0].sum()
    
    return {
        'symbol': symbol,
        'bars': len(df),
        'atr': round(atr, 4),
        'atr_pct': round(atr_pct, 3),
        'daily_range_pct': round(daily_range_pct, 3),
        'total_return': round(returns.sum() * 100, 2),
        'volatility': round(returns.std() * 100, 3),
    }

print("VOLATILITY ANALYSIS: Stocks vs Forex")
print("=" * 70)

stocks = ['SPY', 'AAPL', 'TSLA', 'NVDA']
forex = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']

print("\nSTOCKS:")
stock_data = []
for sym in stocks:
    v = analyze_volatility(sym)
    if v:
        print(f"  {sym:10} ATR%:{v['atr_pct']:6.3f}% | Range:{v['daily_range_pct']:5.3f}% | Vol:{v['volatility']:5.3f}%")
        stock_data.append(v)

print("\nFOREX:")
forex_data = []
for sym in forex:
    v = analyze_volatility(sym)
    if v:
        print(f"  {sym:10} ATR%:{v['atr_pct']:6.3f}% | Range:{v['daily_range_pct']:5.3f}% | Vol:{v['volatility']:5.3f}%")
        forex_data.append(v)

# Averages
if stock_data and forex_data:
    print("\n" + "=" * 70)
    print("AVERAGES:")
    
    avg_stock_atr = np.mean([d['atr_pct'] for d in stock_data])
    avg_forex_atr = np.mean([d['atr_pct'] for d in forex_data])
    
    avg_stock_vol = np.mean([d['volatility'] for d in stock_data])
    avg_forex_vol = np.mean([d['volatility'] for d in forex_data])
    
    print(f"  Stocks:  ATR% = {avg_stock_atr:.3f}%, Volatility = {avg_stock_vol:.3f}%")
    print(f"  Forex:   ATR% = {avg_forex_atr:.3f}%, Volatility = {avg_forex_vol:.3f}%")
    print(f"\n  Forex ATR is {avg_forex_atr/avg_stock_atr:.1f}x smaller than stocks")
    print(f"  Forex Vol is {avg_forex_vol/avg_stock_vol:.1f}x smaller than stocks")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED ADJUSTMENTS FOR FOREX:")
    print("  - Tighter SL/TP (0.5x/1.5x ATR instead of 1.5x/3x)")
    print("  - Smaller position sizes to match stock volatility exposure")
    print("  - Consider pip-based SL/TP (20-30 pips) instead of ATR")
    print("  - Or switch to higher timeframes (4h/1D) for larger swings")

print("\nDone!")
