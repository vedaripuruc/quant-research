#!/usr/bin/env python3
"""
Improved Signals Backtest - Momentum signals with quality filters
=================================================================
Enhancements:
- Trend filter: only take signals aligned with 50-EMA direction
- Volume filter: signal bar volume > 1.5x 20-bar average (CRYPTO ONLY)
- Risk/Reward: 1:3 (TP = 3x SL distance)

NOTE: Volume is valid for CRYPTO (centralized exchanges like Binance).
NEVER use volume for FOREX (OTC, broker-specific garbage).

Features:
- Fetches 30-90 days of 1h data
- Runs signal detection on each bar
- Tracks TP/SL hits
- Calculates comprehensive backtest metrics
- Proper position sizing based on risk management
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
import time

# Import signal detection functions
from momentum_signals import (
    calculate_rsi, calculate_ema, calculate_atr,
    calculate_position_size, ASSETS,
    DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT
)

SCRIPT_DIR = os.path.dirname(__file__)


def fetch_historical_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch historical 1h data from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_data = []
    current_start = start_time
    
    print(f"  Fetching {symbol} ({days} days)...", end=" ")
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            
            # Move to next batch
            last_timestamp = data[-1][0]
            current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if not all_data:
        print("FAILED")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    print(f"{len(df)} bars ✓")
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def detect_signals_at_bar(df: pd.DataFrame, bar_idx: int, symbol: str, config: Dict) -> Optional[Dict]:
    """
    Detect signals at a specific bar using only data available up to that point.
    
    This is the key to proper backtesting - no look-ahead bias.
    """
    # Need at least 50 bars of history
    if bar_idx < 50:
        return None
    
    # Slice data up to and including current bar
    hist = df.iloc[:bar_idx + 1].copy()
    
    # Calculate indicators
    hist['rsi'] = calculate_rsi(hist)
    hist['ema_fast'] = calculate_ema(hist, 9)
    hist['ema_slow'] = calculate_ema(hist, 21)
    hist['atr'] = calculate_atr(hist)
    hist['ema_50'] = calculate_ema(hist, 50)
    hist['vol_avg_20'] = hist['volume'].rolling(window=20).mean()
    
    current = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    price = current['close']
    rsi = current['rsi']
    atr = current['atr']
    ema_fast = current['ema_fast']
    ema_slow = current['ema_slow']
    ema_50 = current['ema_50']
    prev_ema_50 = prev['ema_50']
    vol_avg_20 = current['vol_avg_20']
    
    if pd.isna(rsi) or pd.isna(atr) or pd.isna(ema_50) or pd.isna(prev_ema_50) or pd.isna(vol_avg_20):
        return None

    # Volume filter: signal bar volume > 1.5x 20-bar average
    # Valid for CRYPTO (exchange volume is real). DO NOT use for forex.
    if current['volume'] <= (1.5 * vol_avg_20):
        return None
    
    signals = []
    
    # 1. RSI Reversal
    if prev['rsi'] < 30 and rsi >= 30:
        signals.append({"type": "RSI_REVERSAL", "direction": "LONG", "strength": "STRONG" if rsi < 35 else "MEDIUM"})
    elif prev['rsi'] > 70 and rsi <= 70:
        signals.append({"type": "RSI_REVERSAL", "direction": "SHORT", "strength": "STRONG" if rsi > 65 else "MEDIUM"})
    
    # 2. EMA Cross
    prev_diff = prev['ema_fast'] - prev['ema_slow']
    curr_diff = ema_fast - ema_slow
    
    if prev_diff < 0 and curr_diff > 0:
        signals.append({"type": "EMA_CROSS", "direction": "LONG", "strength": "MEDIUM"})
    elif prev_diff > 0 and curr_diff < 0:
        signals.append({"type": "EMA_CROSS", "direction": "SHORT", "strength": "MEDIUM"})
    
    # 3. Breakout
    lookback = hist.iloc[-24:-1]
    if len(lookback) >= 20:
        range_high = lookback['high'].max()
        range_low = lookback['low'].min()
        range_size = range_high - range_low
        
        if range_size < 2 * atr:
            if current['close'] > range_high:
                signals.append({"type": "BREAKOUT", "direction": "LONG", "strength": "STRONG"})
            elif current['close'] < range_low:
                signals.append({"type": "BREAKOUT", "direction": "SHORT", "strength": "STRONG"})
    
    if not signals:
        return None
    
    # Best signal
    best = max(signals, key=lambda x: 1 if x['strength'] == 'STRONG' else 0)
    direction = best['direction']

    # Trend filter: only take signals aligned with 50-EMA direction (slope)
    if direction == "LONG" and ema_50 <= prev_ema_50:
        return None
    if direction == "SHORT" and ema_50 >= prev_ema_50:
        return None
    
    # CRITICAL: Entry at NEXT bar's open (no look-ahead bias)
    # We can't enter at current bar's close - we'd need to use next bar
    # For backtesting, we'll mark entry at current close but this represents
    # entering at next bar's open
    entry = price
    
    # SL/TP
    sl_distance = atr * 1.5
    tp_distance = sl_distance * 3
    
    if direction == "LONG":
        stop_loss = entry - sl_distance
        take_profit = entry + tp_distance
    else:
        stop_loss = entry + sl_distance
        take_profit = entry - tp_distance
    
    return {
        "symbol": symbol,
        "name": config['name'],
        "timestamp": df.index[bar_idx],
        "bar_idx": bar_idx,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "direction": direction,
        "signal_type": best['type'],
        "atr": atr,
        "rsi": rsi,
        "ema_50": ema_50,
        "volume": current['volume'],
        "vol_avg_20": vol_avg_20,
    }


def simulate_trade(signal: Dict, df: pd.DataFrame, 
                   account_size: float, risk_pct: float) -> Dict:
    """
    Simulate a trade from signal entry to TP/SL.
    
    Checks bars AFTER entry for TP/SL hit.
    """
    entry_idx = signal['bar_idx']
    entry = signal['entry']
    stop_loss = signal['stop_loss']
    take_profit = signal['take_profit']
    direction = signal['direction']
    
    # Position sizing
    position = calculate_position_size(entry, stop_loss, account_size, risk_pct)
    units = position['position_size_units']
    risk_amount = position['risk_amount_usd']
    
    # Check bars AFTER entry
    outcome = "OPEN"
    exit_price = None
    exit_idx = None
    
    for idx in range(entry_idx + 1, len(df)):
        bar = df.iloc[idx]
        high = bar['high']
        low = bar['low']
        
        if direction == "LONG":
            # Check SL first (conservative)
            if low <= stop_loss:
                outcome = "LOSS"
                exit_price = stop_loss
                exit_idx = idx
                break
            elif high >= take_profit:
                outcome = "WIN"
                exit_price = take_profit
                exit_idx = idx
                break
        else:  # SHORT
            if high >= stop_loss:
                outcome = "LOSS"
                exit_price = stop_loss
                exit_idx = idx
                break
            elif low <= take_profit:
                outcome = "WIN"
                exit_price = take_profit
                exit_idx = idx
                break
    
    # Calculate P&L
    if outcome == "WIN":
        pnl = risk_amount * 3  # 1:3 RR
        r_multiple = 3.0
    elif outcome == "LOSS":
        pnl = -risk_amount
        r_multiple = -1.0
    else:  # Still open at end of data
        # Calculate unrealized at last bar
        last_price = df.iloc[-1]['close']
        if direction == "LONG":
            pnl = (last_price - entry) * units
        else:
            pnl = (entry - last_price) * units
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0
    
    bars_held = (exit_idx - entry_idx) if exit_idx else (len(df) - entry_idx)
    
    return {
        **signal,
        "outcome": outcome,
        "exit_price": exit_price,
        "exit_idx": exit_idx,
        "exit_time": df.index[exit_idx] if exit_idx else None,
        "position_size_units": units,
        "position_size_usd": position['position_size_usd'],
        "risk_amount_usd": risk_amount,
        "pnl_usd": round(pnl, 2),
        "r_multiple": round(r_multiple, 2),
        "bars_held": bars_held,
        "hours_held": bars_held,  # 1h bars
    }


def calculate_drawdown(pnl_series: List[float], account_size: float) -> Tuple[float, float]:
    """Calculate max drawdown from P&L series relative to account equity."""
    if not pnl_series:
        return 0, 0
    
    # Equity curve starting from account_size
    cumulative = np.cumsum(pnl_series)
    equity = account_size + cumulative
    
    # Peak equity at each point
    running_max = np.maximum.accumulate(equity)
    
    # Drawdown = peak - current equity
    drawdowns = running_max - equity
    
    max_dd = np.max(drawdowns)
    peak_at_max_dd = running_max[np.argmax(drawdowns)]
    max_dd_pct = (max_dd / peak_at_max_dd) * 100 if peak_at_max_dd > 0 else 0
    
    return round(max_dd, 2), round(max_dd_pct, 2)


def run_backtest(symbols: List[str] = None, days: int = 90,
                 account_size: float = DEFAULT_ACCOUNT_SIZE,
                 risk_pct: float = DEFAULT_RISK_PCT,
                 cooldown_bars: int = 6) -> Dict:
    """
    Run full backtest.
    
    Args:
        symbols: List of symbols to test (defaults to all ASSETS)
        days: Days of history to fetch
        account_size: Account size for position sizing
        risk_pct: Risk per trade
        cooldown_bars: Minimum bars between signals for same asset
    
    Returns:
        Backtest results dict
    """
    if symbols is None:
        symbols = list(ASSETS.keys())
    
    print("="*60)
    print("BACKTEST - IMPROVED MOMENTUM SIGNALS")
    print("="*60)
    print(f"Period: {days} days")
    print(f"Account: ${account_size:,} | Risk: {risk_pct*100:.1f}% per trade")
    print(f"Symbols: {', '.join(symbols)}")
    print("-"*60)
    
    # Fetch data
    print("\n📥 Fetching historical data...")
    data = {}
    for symbol in symbols:
        df = fetch_historical_data(symbol, days)
        if not df.empty:
            data[symbol] = df
    
    if not data:
        print("❌ No data fetched")
        return {"error": "No data"}
    
    # Run signal detection
    print("\n🔍 Detecting signals...")
    all_trades = []
    
    for symbol, df in data.items():
        config = ASSETS[symbol]
        last_signal_bar = -cooldown_bars  # Allow first signal
        
        for bar_idx in range(50, len(df)):
            # Cooldown check
            if bar_idx - last_signal_bar < cooldown_bars:
                continue
            
            signal = detect_signals_at_bar(df, bar_idx, symbol, config)
            
            if signal:
                trade = simulate_trade(signal, df, account_size, risk_pct)
                all_trades.append(trade)
                last_signal_bar = bar_idx
    
    # Sort by timestamp
    all_trades.sort(key=lambda x: x['timestamp'])
    
    print(f"  Found {len(all_trades)} signals")
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    
    closed = [t for t in all_trades if t['outcome'] in ['WIN', 'LOSS']]
    wins = [t for t in closed if t['outcome'] == 'WIN']
    losses = [t for t in closed if t['outcome'] == 'LOSS']
    
    if not closed:
        print("❌ No closed trades")
        return {"error": "No closed trades", "total_signals": len(all_trades)}
    
    # P&L series for drawdown
    pnl_series = [t['pnl_usd'] for t in closed]
    max_dd, max_dd_pct = calculate_drawdown(pnl_series, account_size)
    
    # Gross profit/loss
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))
    
    # R-multiples
    r_multiples = [t['r_multiple'] for t in closed]
    avg_r = sum(r_multiples) / len(r_multiples)
    
    # By signal type
    signal_types = {}
    for t in closed:
        st = t['signal_type']
        if st not in signal_types:
            signal_types[st] = {'wins': 0, 'losses': 0, 'pnl': 0}
        if t['outcome'] == 'WIN':
            signal_types[st]['wins'] += 1
        else:
            signal_types[st]['losses'] += 1
        signal_types[st]['pnl'] += t['pnl_usd']
    
    # By symbol
    symbol_stats = {}
    for t in closed:
        sym = t['symbol']
        if sym not in symbol_stats:
            symbol_stats[sym] = {'wins': 0, 'losses': 0, 'pnl': 0}
        if t['outcome'] == 'WIN':
            symbol_stats[sym]['wins'] += 1
        else:
            symbol_stats[sym]['losses'] += 1
        symbol_stats[sym]['pnl'] += t['pnl_usd']
    
    results = {
        "period_days": days,
        "account_size": account_size,
        "risk_per_trade": risk_pct * 100,
        "symbols": symbols,
        "total_signals": len(all_trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl_usd": round(sum(pnl_series), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        "avg_r": round(avg_r, 2),
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "avg_win_usd": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss_usd": round(gross_loss / len(losses), 2) if losses else 0,
        "avg_bars_held": round(sum(t['bars_held'] for t in closed) / len(closed), 1),
        "expectancy_r": round(avg_r, 2),
        "expectancy_usd": round(sum(pnl_series) / len(closed), 2),
        "return_pct": round(sum(pnl_series) / account_size * 100, 2),
        "by_signal_type": signal_types,
        "by_symbol": symbol_stats,
        "trades": closed,  # Include all trades for detailed analysis
    }
    
    return results


def print_results(results: Dict):
    """Pretty print backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"""
📅 Period:        {results['period_days']} days
💰 Account:       ${results['account_size']:,}
⚠️  Risk/Trade:    {results['risk_per_trade']}%

📊 PERFORMANCE:
├─ Total Signals: {results['total_signals']}
├─ Closed:        {results['closed_trades']}
│  ├─ Wins:       {results['wins']}
│  └─ Losses:     {results['losses']}
├─ Win Rate:      {results['win_rate']}%
├─ Avg R:         {results['avg_r']}
└─ Profit Factor: {results['profit_factor']}

💵 FINANCIALS:
├─ Total P&L:     ${results['total_pnl_usd']:+,.2f}
├─ Gross Profit:  ${results['gross_profit']:,.2f}
├─ Gross Loss:    ${results['gross_loss']:,.2f}
├─ Avg Win:       ${results['avg_win_usd']:,.2f}
├─ Avg Loss:      ${results['avg_loss_usd']:,.2f}
├─ Expectancy:    ${results['expectancy_usd']:+,.2f}/trade
└─ Return:        {results['return_pct']:+.2f}%

📉 RISK:
├─ Max Drawdown:  ${results['max_drawdown_usd']:,.2f}
└─ Max DD %:      {results['max_drawdown_pct']:.2f}%

⏱️  TIMING:
└─ Avg Bars Held: {results['avg_bars_held']} hours
""")
    
    # By signal type
    print("📈 BY SIGNAL TYPE:")
    for sig_type, stats in results['by_signal_type'].items():
        total = stats['wins'] + stats['losses']
        wr = round(stats['wins'] / total * 100, 1) if total > 0 else 0
        print(f"  {sig_type}: {stats['wins']}W/{stats['losses']}L ({wr}%) | P&L: ${stats['pnl']:+,.2f}")
    
    # By symbol
    print("\n🪙 BY SYMBOL:")
    for symbol, stats in results['by_symbol'].items():
        total = stats['wins'] + stats['losses']
        wr = round(stats['wins'] / total * 100, 1) if total > 0 else 0
        name = ASSETS[symbol]['name']
        print(f"  {name}: {stats['wins']}W/{stats['losses']}L ({wr}%) | P&L: ${stats['pnl']:+,.2f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest momentum signals")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--account", type=float, default=100000, help="Account size (default: 100000)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk %% per trade (default: 1.0)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = run_backtest(
        days=args.days,
        account_size=args.account,
        risk_pct=args.risk / 100
    )
    
    print_results(results)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(SCRIPT_DIR, "improved_backtest_results.json")
    
    # Don't save full trade list to file (too large)
    save_results = {k: v for k, v in results.items() if k != 'trades'}
    save_results['trade_count'] = len(results.get('trades', []))
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()
