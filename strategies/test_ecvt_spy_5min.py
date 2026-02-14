"""
ECVT on SPY 5-min + QQQ/IWM hourly — extended equity tests.
yfinance limits 5-min data to 60 days.
"""
import sys
sys.path.insert(0, '/home/sobranceiro/.openclaw/workspace/projects/strat-research')

import numpy as np
import pandas as pd
import yfinance as yf
from ecvt_fast import ECVTParams, generate_signals, run_backtest, print_results
from datetime import datetime, timedelta


def test_ticker_hourly(symbol: str, years: int = 2):
    """Test ECVT on any ticker with hourly data."""
    ticker = yf.Ticker(symbol)
    end = datetime.now()
    start = end - timedelta(days=min(years * 365, 729))
    df = ticker.history(start=start, end=end, interval="1h").reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    
    print(f"\n{symbol}: {len(df)} bars, vol_mean={df['volume'].mean():.0f}, zero_vol={((df['volume']==0).mean()*100):.1f}%")
    
    params = ECVTParams()
    sig_df = generate_signals(df.copy(), params)
    result = run_backtest(sig_df, params)
    print_results(result, f"ECVT {symbol} Hourly")
    
    # Also no_vol_filter
    params2 = ECVTParams(volume_percentile=50.0)
    sig_df2 = generate_signals(df.copy(), params2)
    result2 = run_backtest(sig_df2, params2)
    s = "+" if result2.total_pnl_bps >= 0 else ""
    print(f"  no_vol_filter: {result2.num_trades} trades, {s}{result2.total_pnl_bps:.1f} bps, WR {result2.win_rate:.1%}")
    
    return result


def test_spy_5min():
    """Test SPY on 5-min bars (limited to ~60 days by yfinance)."""
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="60d", interval="5m").reset_index()
    df.columns = [c.lower() for c in df.columns]
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    
    print(f"\nSPY 5-min: {len(df)} bars")
    
    # Adapt params for 5-min (more bars per day, tighter moves)
    params = ECVTParams(
        entropy_window=120,       # ~10 hours
        entropy_lookback=500,
        min_trail_return=0.0003,  # lower for 5-min moves
        max_trail_return=0.002,
        stop_loss_pct=0.001,
        take_profit_pct=0.003,
        timeout_bars=78,          # ~1 trading day
        cost_per_trade_bps=1.0,
    )
    
    sig_df = generate_signals(df.copy(), params)
    result = run_backtest(sig_df, params)
    print_results(result, "ECVT SPY 5-min (60d)")
    return result


def main():
    print("=" * 70)
    print("  Extended ECVT Equity Tests")
    print("=" * 70)
    
    # SPY 5-min
    test_spy_5min()
    
    # Multiple tickers hourly
    for sym in ['QQQ', 'IWM', 'GLD', 'TLT']:
        test_ticker_hourly(sym)


if __name__ == "__main__":
    main()
