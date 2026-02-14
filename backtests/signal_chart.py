#!/usr/bin/env python3
"""
Signal Chart Generator - Plot trades with entry, SL, TP
Sends charts via email
"""
import json
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import subprocess
import sys

# Config
HISTORY_FILE = Path(__file__).parent / "signal_history.json"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

TICKER_MAP = {
    "Chainlink": "LINK-USD",
    "LINK": "LINK-USD", 
    "Cardano": "ADA-USD",
    "ADA": "ADA-USD",
    "XRP": "XRP-USD",
    "Bitcoin": "BTC-USD",
    "BTC": "BTC-USD",
    "Ethereum": "ETH-USD",
    "ETH": "ETH-USD",
}

def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"signals": []}

def generate_chart(signal: dict, hours_before: int = 4, hours_after: int = 4) -> Path:
    """Generate a chart for a signal with entry, SL, TP lines and entry time marker"""
    
    asset = signal["asset"]
    ticker = TICKER_MAP.get(asset, f"{asset}-USD")
    
    # Parse signal timestamp
    signal_time = datetime.fromisoformat(signal["timestamp"].replace('Z', '+00:00'))
    if signal_time.tzinfo is None:
        signal_time = signal_time.replace(tzinfo=None)
    
    # Get price data around the signal time
    start_time = signal_time - timedelta(hours=hours_before)
    end_time = signal_time + timedelta(hours=hours_after)
    
    # If end_time is in the future, use now
    if end_time > datetime.now():
        end_time = datetime.now()
    
    data = yf.Ticker(ticker)
    df = data.history(start=start_time, end=end_time, interval="5m")
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Make index timezone-naive for comparison
    df.index = df.index.tz_localize(None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Plot candlesticks (simplified as line with fill)
    ax.fill_between(df.index, df['Low'], df['High'], alpha=0.3, color='#4a4a6a')
    ax.plot(df.index, df['Close'], color='#00d4ff', linewidth=1.5, label='Price')
    
    # Entry, SL, TP lines
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    direction = signal["direction"].upper()
    
    # ENTRY TIME - vertical line (yellow/gold)
    signal_time_naive = signal_time.replace(tzinfo=None) if signal_time.tzinfo else signal_time
    ax.axvline(x=signal_time_naive, color='#ffd700', linestyle='-', linewidth=3, alpha=0.8, label='Entry Time')
    
    # Find the actual price at entry time (closest candle)
    time_diffs = abs(df.index - signal_time_naive)
    closest_idx = time_diffs.argmin()
    actual_price_at_entry = df['Close'].iloc[closest_idx]
    
    # Add entry marker ON THE PRICE LINE at entry time
    ax.scatter([signal_time_naive], [actual_price_at_entry], color='#ffd700', s=400, zorder=10, marker='o', edgecolors='white', linewidths=3)
    ax.annotate(f'ENTRY\n${entry} (actual: ${actual_price_at_entry:.2f})', 
               xy=(signal_time_naive, actual_price_at_entry),
               xytext=(-80, -50), textcoords='offset points',
               fontsize=10, color='#ffd700', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#ffd700', lw=2),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='#ffd700'))
    
    # Entry price line (white, dotted)
    ax.axhline(y=entry, color='#ffffff', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # SL line (red)
    ax.axhline(y=sl, color='#ff4757', linestyle='--', linewidth=2, label=f'SL: ${sl}')
    
    # TP line (green)
    ax.axhline(y=tp, color='#2ed573', linestyle='--', linewidth=2, label=f'TP: ${tp}')
    
    # Fill zones only AFTER entry time
    after_entry = df.index >= signal_time_naive
    if direction == "LONG":
        ax.fill_between(df.index[after_entry], entry, tp, alpha=0.15, color='#2ed573')  # Profit zone
        ax.fill_between(df.index[after_entry], sl, entry, alpha=0.15, color='#ff4757')  # Loss zone
    else:  # SHORT
        ax.fill_between(df.index[after_entry], tp, entry, alpha=0.15, color='#2ed573')  # Profit zone
        ax.fill_between(df.index[after_entry], entry, sl, alpha=0.15, color='#ff4757')  # Loss zone
    
    # Mark outcome if closed
    if signal.get("status") == "closed" and signal.get("exit_price"):
        exit_price = signal["exit_price"]
        exit_time_str = signal.get("exit_time")
        outcome = signal.get("outcome", "")
        color = '#2ed573' if outcome == "TP" else '#ff4757'
        
        # Find when SL/TP was hit in the data
        if exit_time_str:
            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
            exit_time = exit_time.replace(tzinfo=None) if exit_time.tzinfo else exit_time
        else:
            # Find the candle where SL/TP was breached
            if direction == "LONG":
                if outcome == "SL":
                    hit_idx = df[df['Low'] <= sl].index
                else:
                    hit_idx = df[df['High'] >= tp].index
            else:  # SHORT
                if outcome == "SL":
                    hit_idx = df[df['High'] >= sl].index
                else:
                    hit_idx = df[df['Low'] <= tp].index
            
            exit_time = hit_idx[0] if len(hit_idx) > 0 else df.index[-1]
        
        ax.scatter([exit_time], [exit_price], color=color, s=300, zorder=10, marker='x', linewidths=4)
        ax.annotate(f'{outcome}\n${exit_price:.2f}', 
                   xy=(exit_time, exit_price),
                   xytext=(40, -30), textcoords='offset points',
                   fontsize=11, color=color, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=2),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=color))
    
    # Styling
    ax.set_title(f"{'🔴 SHORT' if direction == 'SHORT' else '🟢 LONG'} {asset} | Signal #{signal['id']}", 
                 fontsize=16, color='white', fontweight='bold', pad=20)
    
    ax.set_xlabel('Time', color='#888888')
    ax.set_ylabel('Price (USD)', color='#888888')
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#4a4a6a')
    ax.spines['top'].set_color('#4a4a6a')
    ax.spines['left'].set_color('#4a4a6a')
    ax.spines['right'].set_color('#4a4a6a')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    # Legend
    legend = ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#4a4a6a')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add signal info box
    info_text = f"Direction: {direction}\n"
    info_text += f"Entry: ${entry}\n"
    info_text += f"Stop Loss: ${sl}\n"
    info_text += f"Take Profit: ${tp}\n"
    info_text += f"R:R: 1:2\n"
    if signal.get("reason"):
        info_text += f"Reason: {signal['reason'][:30]}..."
    
    props = dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9, edgecolor='#4a4a6a')
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, color='white', family='monospace')
    
    # Status badge
    status = signal.get("status", "open")
    if status == "closed":
        outcome = signal.get("outcome", "")
        pnl = signal.get("pnl_pct", 0)
        badge_color = '#2ed573' if outcome == "TP" else '#ff4757'
        badge_text = f"{'✅ WIN' if outcome == 'TP' else '❌ LOSS'} ({pnl:+.2f}%)"
    else:
        badge_color = '#ffa502'
        badge_text = "📊 OPEN"
    
    ax.text(0.02, 0.98, badge_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=badge_color, alpha=0.9),
            color='white')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = CHARTS_DIR / f"signal_{signal['id']}_{timestamp}.png"
    plt.savefig(chart_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    
    print(f"📊 Chart saved: {chart_path}")
    return chart_path

def send_chart_discord(chart_path: Path, signal: dict):
    """Send chart via Discord DM using openclaw message tool"""
    import json
    
    direction = signal["direction"].upper()
    asset = signal["asset"]
    status = signal.get("status", "open")
    outcome = signal.get("outcome", "")
    pnl = signal.get("pnl_pct", 0)
    
    if status == "closed":
        emoji = "✅" if outcome == "TP" else "❌"
        header = f"{emoji} **{direction} {asset} - {outcome}** ({pnl:+.2f}%)"
    else:
        header = f"🎯 **NEW SIGNAL: {direction} {asset}** @ ${signal['entry']}"
    
    body = f"""{header}

Entry: ${signal['entry']} | SL: ${signal['sl']} | TP: ${signal['tp']}
R:R: 1:2
Reason: {signal.get('reason', 'N/A')}"""
    
    # Use openclaw CLI to send Discord DM with image
    cmd = [
        "openclaw", "message", "send",
        "--channel", "discord",
        "--target", "user:402468422576898049",
        "--message", body,
        "--media", str(chart_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ Discord DM sent with chart")
        else:
            print(f"⚠ Discord send failed: {result.stderr}")
            # Fallback: try without image
            cmd_text = [
                "openclaw", "message", "send",
                "--channel", "discord", 
                "--target", "user:402468422576898049",
                "--message", body + f"\n\n📊 Chart: {chart_path}"
            ]
            subprocess.run(cmd_text, capture_output=True, text=True, timeout=30)
    except Exception as e:
        print(f"⚠ Discord error: {e}")
    
    return chart_path

def chart_signal(signal_id: int = None, send: bool = False):
    """Generate chart for a signal (latest if no ID given)"""
    history = load_history()
    
    if not history["signals"]:
        print("No signals to chart")
        return None
    
    if signal_id:
        signal = next((s for s in history["signals"] if s["id"] == signal_id), None)
        if not signal:
            print(f"Signal #{signal_id} not found")
            return None
    else:
        signal = history["signals"][-1]  # Latest
    
    chart_path = generate_chart(signal)
    
    if send:
        send_chart_discord(chart_path, signal)
    
    return chart_path

def chart_all_open(send: bool = False):
    """Chart all open signals"""
    history = load_history()
    open_signals = [s for s in history["signals"] if s.get("status") == "open"]
    
    if not open_signals:
        print("No open signals")
        return []
    
    paths = []
    for signal in open_signals:
        try:
            path = generate_chart(signal)
            paths.append(path)
            if send:
                send_chart_discord(path, signal)
        except Exception as e:
            print(f"⚠ Error charting signal #{signal['id']}: {e}")
    
    return paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Chart Generator")
    parser.add_argument("command", choices=["chart", "all", "latest"], help="Command")
    parser.add_argument("--id", type=int, help="Signal ID to chart")
    parser.add_argument("--send", action="store_true", help="Send via Discord DM")
    
    args = parser.parse_args()
    
    if args.command == "chart":
        chart_signal(args.id, args.send)
    elif args.command == "all":
        chart_all_open(args.send)
    elif args.command == "latest":
        chart_signal(None, args.send)
