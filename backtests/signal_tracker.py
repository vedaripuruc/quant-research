#!/usr/bin/env python3
"""
Signal Tracker - Track outcomes of generated signals
====================================================
Checks if signals hit TP or SL and logs results.

Features:
- Loads active_signals.json
- Fetches price data since signal entry
- Determines WIN/LOSS/OPEN outcome
- Calculates actual P&L based on position size
- Logs to signals_history.json
- Calculates running statistics
"""

import json
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# File paths
SCRIPT_DIR = os.path.dirname(__file__)
ACTIVE_SIGNALS_FILE = os.path.join(SCRIPT_DIR, "active_signals.json")
HISTORY_FILE = os.path.join(SCRIPT_DIR, "signals_history.json")


def fetch_klines_since(symbol: str, start_time: datetime, 
                       end_time: datetime = None) -> pd.DataFrame:
    """Fetch 1h klines from start_time to now (or end_time)."""
    url = "https://api.binance.com/api/v3/klines"
    
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int((end_time or datetime.now()).timestamp() * 1000)
    
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000  # Max 1000 candles
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close']]
    
    except Exception as e:
        print(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()


def fetch_live_price(symbol: str) -> Optional[float]:
    """Fetch current live price."""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return float(response.json()['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


def check_signal_outcome(signal: Dict) -> Dict:
    """
    Check if a signal hit TP or SL.
    
    Returns updated signal dict with:
    - outcome: WIN/LOSS/OPEN
    - exit_price: Price at which position closed
    - exit_time: When position closed
    - actual_pnl_usd: Actual P&L in USD
    - r_multiple: Profit in terms of risk units
    - time_to_resolution: How long until TP/SL hit
    """
    symbol = signal["symbol"]
    direction = signal["direction"]
    entry = signal["entry"]
    stop_loss = signal["stop_loss"]
    take_profit = signal["take_profit"]
    entry_time = datetime.fromisoformat(signal["timestamp"])
    
    # Get position sizing (use defaults if missing from old signals)
    position_units = signal.get("position_size_units", 0)
    risk_amount = signal.get("risk_amount_usd", 1000)  # Default 1% of 100k
    
    # Fetch price data since entry
    df = fetch_klines_since(symbol, entry_time)
    
    if df.empty:
        return {**signal, "outcome": "OPEN", "check_time": datetime.now().isoformat()}
    
    # CRITICAL: Only check candles AFTER entry time
    df = df[df.index > entry_time]
    
    if df.empty:
        # No candles after entry yet - still open
        current_price = fetch_live_price(symbol)
        return {
            **signal,
            "outcome": "OPEN",
            "current_price": current_price,
            "unrealized_pnl_usd": calculate_pnl(entry, current_price, position_units, direction) if current_price else 0,
            "check_time": datetime.now().isoformat()
        }
    
    # Check each candle for TP/SL hit
    exit_price = None
    exit_time = None
    outcome = "OPEN"
    
    for timestamp, candle in df.iterrows():
        high = candle['high']
        low = candle['low']
        
        if direction == "LONG":
            # Check SL first (conservative - assume worst case)
            if low <= stop_loss:
                exit_price = stop_loss
                exit_time = timestamp
                outcome = "LOSS"
                break
            # Then check TP
            elif high >= take_profit:
                exit_price = take_profit
                exit_time = timestamp
                outcome = "WIN"
                break
        else:  # SHORT
            # Check SL first
            if high >= stop_loss:
                exit_price = stop_loss
                exit_time = timestamp
                outcome = "LOSS"
                break
            # Then check TP
            elif low <= take_profit:
                exit_price = take_profit
                exit_time = timestamp
                outcome = "WIN"
                break
    
    result = {**signal, "outcome": outcome, "check_time": datetime.now().isoformat()}
    
    if outcome == "OPEN":
        # Still open - calculate unrealized P&L
        current_price = fetch_live_price(symbol)
        result["current_price"] = current_price
        result["unrealized_pnl_usd"] = calculate_pnl(entry, current_price, position_units, direction) if current_price else 0
    else:
        # Closed - calculate realized P&L
        result["exit_price"] = exit_price
        result["exit_time"] = exit_time.isoformat()
        result["actual_pnl_usd"] = calculate_pnl(entry, exit_price, position_units, direction)
        
        # R-multiple (how many R we made/lost)
        sl_distance = abs(entry - stop_loss)
        if sl_distance > 0:
            price_move = exit_price - entry if direction == "LONG" else entry - exit_price
            result["r_multiple"] = round(price_move / sl_distance, 2)
        else:
            result["r_multiple"] = 0
        
        # Time to resolution
        result["time_to_resolution_hours"] = round(
            (exit_time - entry_time).total_seconds() / 3600, 1
        )
    
    return result


def calculate_pnl(entry: float, exit: float, units: float, direction: str) -> float:
    """Calculate P&L in USD."""
    if direction == "LONG":
        return round((exit - entry) * units, 2)
    else:  # SHORT
        return round((entry - exit) * units, 2)


def load_history() -> List[Dict]:
    """Load signal history from file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history: List[Dict]):
    """Save signal history to file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def calculate_stats(history: List[Dict]) -> Dict:
    """Calculate performance statistics from history."""
    if not history:
        return {"total_signals": 0}
    
    closed = [h for h in history if h.get("outcome") in ["WIN", "LOSS"]]
    wins = [h for h in closed if h["outcome"] == "WIN"]
    losses = [h for h in closed if h["outcome"] == "LOSS"]
    open_signals = [h for h in history if h.get("outcome") == "OPEN"]
    
    total_pnl = sum(h.get("actual_pnl_usd", 0) for h in closed)
    gross_profit = sum(h.get("actual_pnl_usd", 0) for h in wins)
    gross_loss = abs(sum(h.get("actual_pnl_usd", 0) for h in losses))
    
    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    
    # R-multiples
    r_multiples = [h.get("r_multiple", 0) for h in closed if h.get("r_multiple") is not None]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
    
    # Time to resolution
    resolution_times = [h.get("time_to_resolution_hours", 0) for h in closed]
    avg_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    
    return {
        "total_signals": len(history),
        "closed": len(closed),
        "open": len(open_signals),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "avg_r": round(avg_r, 2),
        "avg_resolution_hours": round(avg_time, 1),
        "expectancy_usd": round((avg_win * len(wins) - avg_loss * len(losses)) / len(closed), 2) if closed else 0,
    }


def update_signals() -> Tuple[List[Dict], Dict]:
    """
    Main function: Update all active signals and return results.
    
    Returns:
        (updated_history, stats)
    """
    history = load_history()
    
    # Load active signals
    if not os.path.exists(ACTIVE_SIGNALS_FILE):
        print("No active signals file found")
        return history, calculate_stats(history)
    
    try:
        with open(ACTIVE_SIGNALS_FILE, 'r') as f:
            active = json.load(f)
    except Exception as e:
        print(f"Error loading active signals: {e}")
        return history, calculate_stats(history)
    
    signals = active.get("signals", [])
    
    if not signals:
        print("No active signals to track")
        return history, calculate_stats(history)
    
    print(f"📊 Checking {len(signals)} active signal(s)...")
    
    # Create a set of existing signal IDs (by timestamp+symbol)
    existing_ids = {(h["timestamp"], h["symbol"]) for h in history}
    
    updated = 0
    new_signals = 0
    
    for signal in signals:
        sig_id = (signal["timestamp"], signal["symbol"])
        
        # Check outcome
        result = check_signal_outcome(signal)
        
        if sig_id in existing_ids:
            # Update existing entry in history
            for i, h in enumerate(history):
                if (h["timestamp"], h["symbol"]) == sig_id:
                    if h.get("outcome") == "OPEN" and result["outcome"] != "OPEN":
                        history[i] = result
                        updated += 1
                        print(f"  ✅ {signal['name']}: {result['outcome']} (R: {result.get('r_multiple', 'N/A')})")
                    elif result["outcome"] == "OPEN":
                        history[i] = result  # Update current price
                    break
        else:
            # New signal
            history.append(result)
            new_signals += 1
            outcome_str = result['outcome']
            if outcome_str == "OPEN":
                print(f"  📝 {signal['name']}: New signal added (OPEN)")
            else:
                print(f"  ✅ {signal['name']}: {outcome_str} (R: {result.get('r_multiple', 'N/A')})")
    
    # Save updated history
    save_history(history)
    
    stats = calculate_stats(history)
    
    print(f"\n📈 Summary: {new_signals} new, {updated} resolved")
    print(f"   Win rate: {stats['win_rate']}% | Total P&L: ${stats['total_pnl_usd']:,.2f}")
    
    return history, stats


def print_stats(stats: Dict):
    """Print formatted statistics."""
    print("\n" + "="*50)
    print("SIGNAL PERFORMANCE STATISTICS")
    print("="*50)
    
    if stats["total_signals"] == 0:
        print("No signals tracked yet.")
        return
    
    print(f"""
Total Signals:    {stats['total_signals']}
├─ Closed:        {stats['closed']}
│  ├─ Wins:       {stats['wins']}
│  └─ Losses:     {stats['losses']}
└─ Open:          {stats['open']}

Win Rate:         {stats['win_rate']}%
Average R:        {stats['avg_r']}
Profit Factor:    {stats['profit_factor']}
Expectancy:       ${stats['expectancy_usd']}/trade

Financials:
├─ Total P&L:     ${stats['total_pnl_usd']:,.2f}
├─ Gross Profit:  ${stats['gross_profit']:,.2f}
├─ Gross Loss:    ${stats['gross_loss']:,.2f}
├─ Avg Win:       ${stats['avg_win_usd']:,.2f}
└─ Avg Loss:      ${stats['avg_loss_usd']:,.2f}

Avg Time to Resolution: {stats['avg_resolution_hours']} hours
""")


def main():
    """Main entry point."""
    print("="*50)
    print("SIGNAL TRACKER")
    print("="*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    history, stats = update_signals()
    print_stats(stats)
    
    # Also show open positions
    open_positions = [h for h in history if h.get("outcome") == "OPEN"]
    if open_positions:
        print("\n📍 OPEN POSITIONS:")
        for pos in open_positions:
            current = pos.get("current_price", 0)
            entry = pos.get("entry", 0)
            unrealized = pos.get("unrealized_pnl_usd", 0)
            direction = pos.get("direction", "?")
            
            emoji = "🟢" if unrealized >= 0 else "🔴"
            print(f"  {emoji} {pos['name']} ({direction}): Entry ${entry:.4f} → Current ${current:.4f} | P&L: ${unrealized:+,.2f}")
    
    print(f"\n✅ History saved to {HISTORY_FILE}")
    
    return history, stats


if __name__ == "__main__":
    main()
