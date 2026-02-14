"""
ECVT on SPY — test entropy collapse signal on equities with real volume.
Downloads 2 years of SPY hourly data, runs full backtest + walk-forward.
"""

import sys
sys.path.insert(0, '/home/sobranceiro/.openclaw/workspace/projects/strat-research')

import numpy as np
import pandas as pd
import yfinance as yf
from ecvt_fast import ECVTParams, generate_signals, run_backtest, print_results, BacktestResult
from datetime import datetime, timedelta


def download_spy_hourly(years: int = 2) -> pd.DataFrame:
    """Download SPY hourly data. yfinance limits hourly to 730 days."""
    ticker = yf.Ticker("SPY")
    end = datetime.now()
    start = end - timedelta(days=min(years * 365, 729))
    
    df = ticker.history(start=start, end=end, interval="1h")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # Rename datetime column
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    
    # Filter to market hours only (already done by yfinance for hourly)
    print(f"Downloaded {len(df)} hourly bars for SPY")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Volume stats: mean={df['volume'].mean():.0f}, zero_pct={((df['volume']==0).mean()*100):.1f}%")
    
    return df


def walk_forward(df: pd.DataFrame, params: ECVTParams, 
                 n_windows: int = 10, train_ratio: float = 0.7) -> dict:
    """Walk-forward validation with fixed params (no optimization)."""
    n = len(df)
    window_size = n // n_windows
    
    results = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        if end - start < 500:
            continue
        
        split = start + int((end - start) * train_ratio)
        
        # Generate signals on full window (entropy needs lookback)
        window_df = df.iloc[start:end].copy().reset_index(drop=True)
        sig_df = generate_signals(window_df, params)
        
        # Only evaluate OOS portion
        oos_start = split - start
        oos_df = sig_df.iloc[oos_start:].copy().reset_index(drop=True)
        
        result = run_backtest(oos_df, params)
        results.append({
            'window': i + 1,
            'oos_bars': len(oos_df),
            'trades': result.num_trades,
            'pnl_bps': result.total_pnl_bps,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
        })
        
        sign = "+" if result.total_pnl_bps >= 0 else ""
        print(f"  Window {i+1:2d}: {result.num_trades:3d} trades, "
              f"{sign}{result.total_pnl_bps:.1f} bps, "
              f"WR {result.win_rate:.1%}, PF {result.profit_factor:.2f}")
    
    if not results:
        return {'windows': 0, 'profitable': 0, 'total_oos_bps': 0, 'total_trades': 0}
    
    profitable = sum(1 for r in results if r['pnl_bps'] > 0)
    total_bps = sum(r['pnl_bps'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    
    return {
        'windows': len(results),
        'profitable': profitable,
        'total_oos_bps': total_bps,
        'total_trades': total_trades,
        'results': results,
    }


def test_param_variations(df: pd.DataFrame):
    """Test a few param variations to check robustness."""
    variations = {
        'default': ECVTParams(),
        'tight_sl': ECVTParams(stop_loss_pct=0.001, take_profit_pct=0.003),
        'wide_sl': ECVTParams(stop_loss_pct=0.003, take_profit_pct=0.008),
        'fast_entropy': ECVTParams(entropy_window=60, entropy_lookback=250),
        'slow_entropy': ECVTParams(entropy_window=240, entropy_lookback=1000),
        'low_cost': ECVTParams(cost_per_trade_bps=1.0),
        'no_vol_filter': ECVTParams(volume_percentile=50.0),
    }
    
    print("\n" + "=" * 70)
    print("  Parameter Robustness Check")
    print("=" * 70)
    
    for name, params in variations.items():
        sig_df = generate_signals(df.copy(), params)
        result = run_backtest(sig_df, params)
        sign = "+" if result.total_pnl_bps >= 0 else ""
        print(f"  {name:20s}: {result.num_trades:4d} trades, "
              f"{sign}{result.total_pnl_bps:7.1f} bps, "
              f"WR {result.win_rate:.1%}, PF {result.profit_factor:.2f}")


def main():
    print("=" * 70)
    print("  ECVT on SPY — Equity Test with Real Volume")
    print("=" * 70)
    
    # Download data
    df = download_spy_hourly(years=2)
    
    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} bars — need at least 1000 for meaningful test")
        return
    
    # Default params (same as EURUSD test)
    params = ECVTParams()
    
    # Full backtest
    print("\n--- Full Backtest (default params) ---")
    sig_df = generate_signals(df.copy(), params)
    result = run_backtest(sig_df, params)
    print_results(result, "ECVT SPY Hourly — Full Period")
    
    # Walk-forward
    print("\n--- Walk-Forward Validation (10 windows, 70/30 split) ---")
    wf = walk_forward(df, params, n_windows=10)
    print(f"\n  Summary: {wf['profitable']}/{wf['windows']} windows profitable")
    print(f"  Total OOS: {wf['total_oos_bps']:+.1f} bps across {wf['total_trades']} trades")
    
    # Also try 20 windows for finer granularity
    print("\n--- Walk-Forward (20 windows) ---")
    wf20 = walk_forward(df, params, n_windows=20)
    print(f"\n  Summary: {wf20['profitable']}/{wf20['windows']} windows profitable")
    print(f"  Total OOS: {wf20['total_oos_bps']:+.1f} bps across {wf20['total_trades']} trades")
    
    # Parameter robustness
    test_param_variations(df)
    
    # Equity-adapted params (SPY has different volatility than forex)
    print("\n--- Equity-Adapted Params ---")
    equity_params = ECVTParams(
        stop_loss_pct=0.002,      # wider for equities
        take_profit_pct=0.006,
        min_trail_return=0.0005,  # lower threshold (SPY less volatile than EURUSD)
        max_trail_return=0.003,
        cost_per_trade_bps=1.0,   # SPY has tighter spreads
    )
    sig_df2 = generate_signals(df.copy(), equity_params)
    result2 = run_backtest(sig_df2, equity_params)
    print_results(result2, "ECVT SPY — Equity-Adapted Params")
    
    # WF on equity params
    print("--- WF on Equity-Adapted Params ---")
    wf_eq = walk_forward(df, equity_params, n_windows=10)
    print(f"\n  Summary: {wf_eq['profitable']}/{wf_eq['windows']} windows profitable")
    print(f"  Total OOS: {wf_eq['total_oos_bps']:+.1f} bps across {wf_eq['total_trades']} trades")
    
    # Save results summary
    print("\n\nDone. Check output above for full results.")


if __name__ == "__main__":
    main()
