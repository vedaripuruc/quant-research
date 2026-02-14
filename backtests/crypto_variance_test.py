"""
Crypto Variance Test - Real Data
================================
Test if crypto's high volatility helps or hurts prop firm challenges.

Hypothesis: Crypto has MORE variance but also MORE daily DD kills.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


def fetch_crypto_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch crypto OHLCV data."""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1h")
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_daily_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily high-low ranges as % of price."""
    df = df.copy()
    df['date'] = df.index.date
    
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    daily['range_pct'] = (daily['high'] - daily['low']) / daily['open'] * 100
    daily['move_pct'] = abs(daily['close'] - daily['open']) / daily['open'] * 100
    
    return daily


def analyze_volatility_profile(symbol: str, name: str) -> Dict:
    """Analyze volatility characteristics of an asset."""
    print(f"\nFetching {name} ({symbol})...")
    df = fetch_crypto_data(symbol, days=180)
    
    if df.empty or len(df) < 100:
        print(f"  No data for {symbol}")
        return None
    
    daily = calculate_daily_ranges(df)
    
    # Key metrics
    avg_daily_range = daily['range_pct'].mean()
    max_daily_range = daily['range_pct'].max()
    avg_daily_move = daily['move_pct'].mean()
    
    # How often does daily range exceed X%?
    exceed_3pct = (daily['range_pct'] > 3).mean() * 100
    exceed_5pct = (daily['range_pct'] > 5).mean() * 100
    exceed_10pct = (daily['range_pct'] > 10).mean() * 100
    
    # Probability of hitting SL in single day with different risk sizes
    # If you have 2% SL and daily range is 4%, high chance of hitting SL
    
    result = {
        "symbol": symbol,
        "name": name,
        "bars": len(df),
        "days": len(daily),
        "avg_daily_range_pct": avg_daily_range,
        "max_daily_range_pct": max_daily_range,
        "avg_daily_move_pct": avg_daily_move,
        "days_exceed_3pct": exceed_3pct,
        "days_exceed_5pct": exceed_5pct,
        "days_exceed_10pct": exceed_10pct,
    }
    
    print(f"  Avg daily range: {avg_daily_range:.2f}%")
    print(f"  Max daily range: {max_daily_range:.2f}%")
    print(f"  Days >5% range: {exceed_5pct:.1f}%")
    
    return result


def simulate_random_entries(df: pd.DataFrame, risk_pct: float = 2.0, rr: float = 2.0, n_trades: int = 100) -> Dict:
    """
    Simulate random entry trades to measure real hit rates.
    
    This is a REALISTIC test of variance play.
    """
    if len(df) < 50:
        return None
    
    results = {
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
        "daily_dd_kills": 0,
        "pnl_total": 0,
    }
    
    # Sample random entry points
    valid_indices = list(range(24, len(df) - 24))  # Need room for ATR and trade
    if len(valid_indices) < n_trades:
        n_trades = len(valid_indices)
    
    entry_indices = np.random.choice(valid_indices, n_trades, replace=False)
    
    for idx in entry_indices:
        entry_bar = df.iloc[idx]
        entry_price = entry_bar['open']
        
        # Calculate ATR from previous 24 bars
        prev_bars = df.iloc[idx-24:idx]
        atr = (prev_bars['high'] - prev_bars['low']).mean()
        
        if atr == 0 or entry_price == 0:
            continue
        
        # Random direction
        direction = np.random.choice(["LONG", "SHORT"])
        
        # Set SL/TP based on ATR
        sl_distance = atr * 1.5
        tp_distance = sl_distance * rr
        
        if direction == "LONG":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        # Walk forward max 24 bars (1 day for hourly)
        future_bars = df.iloc[idx:idx+24]
        trade_pnl = 0
        
        for _, bar in future_bars.iterrows():
            if direction == "LONG":
                if bar['low'] <= sl:
                    trade_pnl = -risk_pct
                    results["losses"] += 1
                    break
                elif bar['high'] >= tp:
                    trade_pnl = risk_pct * rr
                    results["wins"] += 1
                    break
            else:
                if bar['high'] >= sl:
                    trade_pnl = -risk_pct
                    results["losses"] += 1
                    break
                elif bar['low'] <= tp:
                    trade_pnl = risk_pct * rr
                    results["wins"] += 1
                    break
        else:
            # Timeout
            results["timeouts"] += 1
            last_close = future_bars.iloc[-1]['close']
            if direction == "LONG":
                raw_pnl = (last_close - entry_price) / sl_distance
            else:
                raw_pnl = (entry_price - last_close) / sl_distance
            trade_pnl = raw_pnl * risk_pct
        
        results["pnl_total"] += trade_pnl
    
    total_trades = results["wins"] + results["losses"] + results["timeouts"]
    
    return {
        "total_trades": total_trades,
        "win_rate": results["wins"] / total_trades * 100 if total_trades > 0 else 0,
        "loss_rate": results["losses"] / total_trades * 100 if total_trades > 0 else 0,
        "timeout_rate": results["timeouts"] / total_trades * 100 if total_trades > 0 else 0,
        "total_pnl": results["pnl_total"],
        "avg_pnl_per_trade": results["pnl_total"] / total_trades if total_trades > 0 else 0
    }


def main():
    print("="*70)
    print("CRYPTO VS FOREX - REAL DATA VARIANCE ANALYSIS")
    print("="*70)
    
    # Assets to test
    assets = [
        ("BTC-USD", "Bitcoin"),
        ("ETH-USD", "Ethereum"),
        ("SOL-USD", "Solana"),
        ("DOGE-USD", "Dogecoin"),
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
        ("GC=F", "Gold"),
    ]
    
    # Part 1: Volatility Profile
    print("\n" + "="*70)
    print("PART 1: VOLATILITY PROFILES")
    print("="*70)
    
    vol_results = []
    for symbol, name in assets:
        result = analyze_volatility_profile(symbol, name)
        if result:
            vol_results.append(result)
    
    # Sort by avg daily range
    vol_results.sort(key=lambda x: x["avg_daily_range_pct"], reverse=True)
    
    print("\n" + "-"*70)
    print(f"{'Asset':<15} {'Avg Range%':<12} {'Max Range%':<12} {'Days>5%':<10}")
    print("-"*70)
    for r in vol_results:
        print(f"{r['name']:<15} {r['avg_daily_range_pct']:<12.2f} "
              f"{r['max_daily_range_pct']:<12.2f} {r['days_exceed_5pct']:<10.1f}%")
    
    # Part 2: Random Entry Simulation
    print("\n" + "="*70)
    print("PART 2: RANDOM ENTRY SIMULATION (2% risk, 1:2 RR)")
    print("="*70)
    
    sim_results = []
    for symbol, name in assets:
        print(f"\nSimulating {name}...")
        df = fetch_crypto_data(symbol, days=180)
        
        if df.empty or len(df) < 100:
            continue
        
        result = simulate_random_entries(df, risk_pct=2.0, rr=2.0, n_trades=200)
        if result:
            result["name"] = name
            sim_results.append(result)
            print(f"  WR: {result['win_rate']:.1f}% | "
                  f"Timeout: {result['timeout_rate']:.1f}% | "
                  f"PnL: {result['total_pnl']:+.1f}%")
    
    print("\n" + "-"*70)
    print(f"{'Asset':<15} {'WR%':<8} {'Loss%':<8} {'Timeout%':<10} {'Total PnL%':<12}")
    print("-"*70)
    for r in sim_results:
        print(f"{r['name']:<15} {r['win_rate']:<8.1f} {r['loss_rate']:<8.1f} "
              f"{r['timeout_rate']:<10.1f} {r['total_pnl']:<12.1f}")
    
    # Part 3: Recommendation
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Find best performers
    best_by_pnl = max(sim_results, key=lambda x: x["total_pnl"]) if sim_results else None
    best_by_wr = max(sim_results, key=lambda x: x["win_rate"]) if sim_results else None
    
    if best_by_pnl:
        print(f"""
BEST BY PnL: {best_by_pnl['name']}
  - Total PnL: {best_by_pnl['total_pnl']:+.1f}%
  - Win Rate: {best_by_pnl['win_rate']:.1f}%

BEST BY WR: {best_by_wr['name']}
  - Win Rate: {best_by_wr['win_rate']:.1f}%
  - Total PnL: {best_by_wr['total_pnl']:+.1f}%

KEY INSIGHT:
Higher volatility doesn't mean better for prop firm challenges.
What matters is:
1. Consistent directional moves (not whipsaw)
2. Reasonable daily ranges (avoid 5% daily DD)
3. Clear trend/momentum patterns
        """)
    
    # Save results
    output = {
        "volatility_profiles": vol_results,
        "simulation_results": sim_results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("crypto_variance_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("✅ Results saved to crypto_variance_results.json")


if __name__ == "__main__":
    main()
