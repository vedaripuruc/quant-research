#!/usr/bin/env python3
"""
Trade Tracker - Forward Testing System
---------------------------------------
Tracks signal entries/exits as if trades were taken.
Stores trade log in JSON for analysis.

Run: python trade_tracker.py --add <signal_file>
     python trade_tracker.py --check
     python trade_tracker.py --report
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Try to import yfinance for price checking
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

TRADES_FILE = Path(__file__).parent / 'trade_log.json'


def load_trades() -> List[Dict]:
    """Load existing trades from file."""
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades: List[Dict]):
    """Save trades to file."""
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)


def add_trade(signal: Dict) -> Dict:
    """Add a new trade from a signal."""
    trade = {
        'id': len(load_trades()) + 1,
        'symbol': signal.get('symbol'),
        'direction': signal.get('type', signal.get('direction', '')).lower(),
        'strategy': signal.get('strategy'),
        'entry_price': signal.get('entry'),
        'stop_loss': signal.get('stop_loss'),
        'take_profit': signal.get('take_profit'),
        'entry_time': datetime.now().isoformat(),
        'signal_time': signal.get('scanned_at', signal.get('time')),
        'status': 'open',
        'exit_price': None,
        'exit_time': None,
        'exit_reason': None,
        'pnl_pct': None,
        'notes': signal.get('notes', ''),
    }
    
    trades = load_trades()
    trades.append(trade)
    save_trades(trades)
    
    return trade


def check_trades() -> List[Dict]:
    """Check open trades against current prices and update status."""
    if not HAS_YF:
        print("Warning: yfinance not installed, cannot check live prices")
        return []
    
    trades = load_trades()
    updated = []
    
    for trade in trades:
        if trade['status'] != 'open':
            continue
        
        symbol = trade['symbol']
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1h')
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            high = hist['High'].iloc[-1]
            low = hist['Low'].iloc[-1]
            
            sl = trade['stop_loss']
            tp = trade['take_profit']
            direction = trade['direction']
            
            # Check if SL or TP hit
            if direction == 'long':
                if low <= sl:
                    trade['status'] = 'closed'
                    trade['exit_price'] = sl
                    trade['exit_reason'] = 'stop_loss'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['pnl_pct'] = (sl - trade['entry_price']) / trade['entry_price'] * 100
                elif high >= tp:
                    trade['status'] = 'closed'
                    trade['exit_price'] = tp
                    trade['exit_reason'] = 'take_profit'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['pnl_pct'] = (tp - trade['entry_price']) / trade['entry_price'] * 100
            else:  # short
                if high >= sl:
                    trade['status'] = 'closed'
                    trade['exit_price'] = sl
                    trade['exit_reason'] = 'stop_loss'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['pnl_pct'] = (trade['entry_price'] - sl) / trade['entry_price'] * 100
                elif low <= tp:
                    trade['status'] = 'closed'
                    trade['exit_price'] = tp
                    trade['exit_reason'] = 'take_profit'
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['pnl_pct'] = (trade['entry_price'] - tp) / trade['entry_price'] * 100
            
            if trade['status'] == 'closed':
                updated.append(trade)
                
        except Exception as e:
            print(f"Error checking {symbol}: {e}")
    
    save_trades(trades)
    return updated


def generate_report() -> str:
    """Generate a performance report."""
    trades = load_trades()
    
    if not trades:
        return "No trades recorded yet."
    
    open_trades = [t for t in trades if t['status'] == 'open']
    closed_trades = [t for t in trades if t['status'] == 'closed']
    
    wins = [t for t in closed_trades if t.get('pnl_pct', 0) > 0]
    losses = [t for t in closed_trades if t.get('pnl_pct', 0) < 0]
    
    total_pnl = sum(t.get('pnl_pct', 0) for t in closed_trades)
    avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    
    report = f"""
================================================================================
TRADE TRACKER REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

SUMMARY
-------
Total Trades: {len(trades)}
Open Trades: {len(open_trades)}
Closed Trades: {len(closed_trades)}

PERFORMANCE (Closed Trades)
---------------------------
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {win_rate:.1f}%
Total P&L: {total_pnl:+.2f}%
Avg Win: {avg_win:+.2f}%
Avg Loss: {avg_loss:+.2f}%

OPEN POSITIONS
--------------
"""
    
    for t in open_trades:
        report += f"  {t['symbol']} {t['direction'].upper()} @ {t['entry_price']:.4f} | SL: {t['stop_loss']:.4f} | TP: {t['take_profit']:.4f}\n"
    
    if not open_trades:
        report += "  (none)\n"
    
    report += "\nRECENT CLOSED TRADES\n--------------------\n"
    
    for t in closed_trades[-5:]:
        emoji = "✅" if t.get('pnl_pct', 0) > 0 else "❌"
        report += f"  {emoji} {t['symbol']} {t['direction'].upper()} | {t['exit_reason']} | P&L: {t.get('pnl_pct', 0):+.2f}%\n"
    
    if not closed_trades:
        report += "  (none)\n"
    
    return report


def add_signals_from_state():
    """Add all signals from signal_state.json as trades."""
    state_file = Path(__file__).parent / 'signal_state.json'
    if not state_file.exists():
        print("No signal_state.json found")
        return
    
    with open(state_file) as f:
        state = json.load(f)
    
    signals = state.get('signals', [])
    existing = load_trades()
    existing_ids = set(f"{t['symbol']}_{t['strategy']}_{t.get('signal_time', '')}" for t in existing)
    
    added = 0
    for signal in signals:
        sig_id = f"{signal['symbol']}_{signal['strategy']}_{signal.get('scanned_at', '')}"
        if sig_id not in existing_ids:
            trade = add_trade(signal)
            print(f"Added: {trade['symbol']} {trade['direction'].upper()} ({trade['strategy']})")
            added += 1
    
    print(f"\nAdded {added} new trades from signals")


def main():
    parser = argparse.ArgumentParser(description='Trade Tracker for Forward Testing')
    parser.add_argument('--add', action='store_true', help='Add trades from signal_state.json')
    parser.add_argument('--check', action='store_true', help='Check open trades against live prices')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    
    args = parser.parse_args()
    
    if args.add:
        add_signals_from_state()
    elif args.check:
        updated = check_trades()
        if updated:
            print(f"Updated {len(updated)} trades:")
            for t in updated:
                emoji = "✅" if t.get('pnl_pct', 0) > 0 else "❌"
                print(f"  {emoji} {t['symbol']} {t['exit_reason']} | P&L: {t.get('pnl_pct', 0):+.2f}%")
        else:
            print("No trades closed")
    elif args.report:
        print(generate_report())
    else:
        # Default: show report
        print(generate_report())


if __name__ == '__main__':
    main()
