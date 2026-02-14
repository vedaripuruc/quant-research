"""
Walk-Forward Blind Test - HOURLY Data
=====================================
Using Binance API for crypto (free, no key needed, years of 1h data)
Using yfinance for Gold/Forex (limited but still useful)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def fetch_binance_klines(symbol: str, interval: str = "1h", days: int = 730) -> pd.DataFrame:
    """
    Fetch historical klines from Binance (free, no API key).
    Max 1000 candles per request, so we paginate.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    print(f"  Fetching {symbol} from Binance ({days} days of {interval} data)...")
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Move start time to after last candle
            current_start = klines[-1][0] + 1
            
            # Rate limit protection
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    Error fetching {symbol}: {e}")
            break
    
    if not all_klines:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"    Got {len(df)} hourly bars from {df.index[0]} to {df.index[-1]}")
    return df


def fetch_yfinance_hourly(symbol: str) -> pd.DataFrame:
    """Fetch hourly data from yfinance (limited to 730 days)."""
    import yfinance as yf
    
    print(f"  Fetching {symbol} from Yahoo (max 730 days 1h)...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="730d", interval="1h")
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    print(f"    Got {len(df)} hourly bars from {df.index[0]} to {df.index[-1]}")
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_max_dd(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown."""
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def simulate_trades(
    df: pd.DataFrame,
    risk_pct: float = 2.0,
    rr: float = 2.0,
    n_trades: int = 100,
    max_bars: int = 24  # 24 hours max hold
) -> Dict:
    """Simulate random entry trades on hourly data."""
    if len(df) < max_bars * 3:
        return None
    
    results = {"equity_curve": [100.0], "wins": 0, "losses": 0, "timeouts": 0}
    
    # Need ATR lookback (24 bars) + trade duration
    valid_idx = list(range(48, len(df) - max_bars))
    if len(valid_idx) < n_trades:
        n_trades = len(valid_idx)
    if n_trades < 20:
        return None
    
    entries = sorted(np.random.choice(valid_idx, n_trades, replace=False))
    equity = 100.0
    
    for idx in entries:
        entry_price = df.iloc[idx]['open']
        
        # ATR from last 24 bars
        prev = df.iloc[idx-24:idx]
        atr = (prev['high'] - prev['low']).mean()
        
        if atr == 0 or entry_price == 0:
            continue
        
        direction = np.random.choice(["LONG", "SHORT"])
        sl_dist = atr * 1.5
        tp_dist = sl_dist * rr
        
        if direction == "LONG":
            sl, tp = entry_price - sl_dist, entry_price + tp_dist
        else:
            sl, tp = entry_price + sl_dist, entry_price - tp_dist
        
        # Walk forward
        future = df.iloc[idx:idx+max_bars]
        pnl = 0
        
        for _, bar in future.iterrows():
            if direction == "LONG":
                if bar['low'] <= sl:
                    pnl = -risk_pct
                    results["losses"] += 1
                    break
                elif bar['high'] >= tp:
                    pnl = risk_pct * rr
                    results["wins"] += 1
                    break
            else:
                if bar['high'] >= sl:
                    pnl = -risk_pct
                    results["losses"] += 1
                    break
                elif bar['low'] <= tp:
                    pnl = risk_pct * rr
                    results["wins"] += 1
                    break
        else:
            results["timeouts"] += 1
            last = future.iloc[-1]['close']
            if direction == "LONG":
                pnl = (last - entry_price) / sl_dist * risk_pct
            else:
                pnl = (entry_price - last) / sl_dist * risk_pct
        
        equity *= (1 + pnl / 100)
        results["equity_curve"].append(equity)
    
    total = results["wins"] + results["losses"] + results["timeouts"]
    if total == 0:
        return None
    
    return {
        "total_trades": total,
        "wins": results["wins"],
        "losses": results["losses"],
        "timeouts": results["timeouts"],
        "win_rate": results["wins"] / total * 100,
        "final_equity": equity,
        "total_return": equity - 100,
        "max_dd": calculate_max_dd(results["equity_curve"]),
        "equity_curve": results["equity_curve"]
    }


def walk_forward_test(
    df: pd.DataFrame,
    name: str,
    n_windows: int = 6,
    train_bars: int = 2160,  # 90 days * 24h
    test_bars: int = 720,    # 30 days * 24h
    trades_per_window: int = 100
) -> Dict:
    """Walk-forward analysis with train/test splits on hourly data."""
    
    if df.empty or len(df) < train_bars + test_bars:
        print(f"  Insufficient data for {name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {name} ({len(df)} hourly bars)")
    print(f"{'='*60}")
    
    results = {
        "name": name,
        "total_bars": len(df),
        "windows": [],
        "in_sample": [],
        "out_of_sample": [],
        "equity_curves_oos": []
    }
    
    total_bars = len(df)
    window_size = train_bars + test_bars
    step = max((total_bars - window_size) // max(n_windows - 1, 1), test_bars)
    
    actual_windows = 0
    for i in range(n_windows):
        start = i * step
        if start + window_size > total_bars:
            break
        
        train_df = df.iloc[start:start+train_bars]
        test_df = df.iloc[start+train_bars:start+window_size]
        
        train_start = train_df.index[0].strftime('%Y-%m-%d')
        train_end = train_df.index[-1].strftime('%Y-%m-%d')
        test_start = test_df.index[0].strftime('%Y-%m-%d')
        test_end = test_df.index[-1].strftime('%Y-%m-%d')
        
        print(f"\n  Window {i+1}:")
        print(f"    Train: {train_start} → {train_end} ({len(train_df)} bars)")
        print(f"    Test:  {test_start} → {test_end} ({len(test_df)} bars)")
        
        train_res = simulate_trades(train_df, n_trades=trades_per_window)
        test_res = simulate_trades(test_df, n_trades=trades_per_window)
        
        if train_res and test_res:
            actual_windows += 1
            results["windows"].append({
                "window": actual_windows,
                "train_period": f"{train_start} → {train_end}",
                "test_period": f"{test_start} → {test_end}",
                "train": {"wr": train_res["win_rate"], "ret": train_res["total_return"], "dd": train_res["max_dd"], "trades": train_res["total_trades"]},
                "test": {"wr": test_res["win_rate"], "ret": test_res["total_return"], "dd": test_res["max_dd"], "trades": test_res["total_trades"]}
            })
            results["in_sample"].append(train_res["total_return"])
            results["out_of_sample"].append(test_res["total_return"])
            results["equity_curves_oos"].append(test_res["equity_curve"])
            
            print(f"    Train: {train_res['total_trades']} trades, WR={train_res['win_rate']:.0f}%, Ret={train_res['total_return']:+.1f}%")
            print(f"    Test:  {test_res['total_trades']} trades, WR={test_res['win_rate']:.0f}%, Ret={test_res['total_return']:+.1f}%")
    
    if results["windows"]:
        results["avg_is"] = np.mean(results["in_sample"])
        results["avg_oos"] = np.mean(results["out_of_sample"])
        results["consistency"] = sum(1 for x in results["out_of_sample"] if x > 0) / len(results["out_of_sample"]) * 100
        results["total_oos_trades"] = sum(w["test"]["trades"] for w in results["windows"])
    
    return results


def create_report(all_results: List[Dict], output_dir: str):
    """Create visual report."""
    valid = [r for r in all_results if r and r.get("windows")]
    
    if not valid:
        print("No valid results to visualize")
        return None, None
    
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    colors = {'pos': '#00C853', 'neg': '#FF1744', 'train': '#7C4DFF', 'test': '#FF6D00'}
    
    # 1. IS vs OOS comparison
    ax1 = fig.add_subplot(gs[0, :])
    assets = [r['name'] for r in valid]
    is_ret = [r['avg_is'] for r in valid]
    oos_ret = [r['avg_oos'] for r in valid]
    
    x = np.arange(len(assets))
    w = 0.35
    ax1.bar(x - w/2, is_ret, w, label='In-Sample', color=colors['train'], alpha=0.8)
    ax1.bar(x + w/2, oos_ret, w, label='Out-of-Sample', color=colors['test'], alpha=0.8)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, fontsize=9, rotation=15)
    ax1.set_ylabel('Avg Return (%)')
    ax1.set_title('Walk-Forward: In-Sample vs Out-of-Sample Returns (HOURLY DATA)', fontsize=14, fontweight='bold')
    ax1.legend()
    
    for i, (is_v, oos_v) in enumerate(zip(is_ret, oos_ret)):
        ax1.annotate(f'{is_v:+.0f}%', (i-w/2, is_v), ha='center', va='bottom' if is_v > 0 else 'top', fontsize=8)
        ax1.annotate(f'{oos_v:+.0f}%', (i+w/2, oos_v), ha='center', va='bottom' if oos_v > 0 else 'top', fontsize=8)
    
    # 2. OOS Consistency
    ax2 = fig.add_subplot(gs[1, 0])
    cons = [r['consistency'] for r in valid]
    bar_colors = [colors['pos'] if c >= 50 else colors['neg'] for c in cons]
    ax2.bar(assets, cons, color=bar_colors, alpha=0.8)
    ax2.axhline(50, color='gray', ls='--', lw=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('% Windows Profitable')
    ax2.set_title('OOS Consistency (≥50% = Good)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    for i, c in enumerate(cons):
        ax2.annotate(f'{c:.0f}%', (i, c+2), ha='center', fontsize=10, fontweight='bold')
    
    # 3. Total OOS Trades (sample size indicator)
    ax3 = fig.add_subplot(gs[1, 1])
    total_trades = [r['total_oos_trades'] for r in valid]
    ax3.bar(assets, total_trades, color=colors['test'], alpha=0.8)
    ax3.set_ylabel('Total OOS Trades')
    ax3.set_title('Sample Size (More = Better Validation)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    for i, t in enumerate(total_trades):
        ax3.annotate(f'{t}', (i, t+5), ha='center', fontsize=10)
    
    # 4. Win Rate Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    wr_by_asset = {r['name']: [w['test']['wr'] for w in r['windows']] for r in valid}
    positions = []
    for i, (name, wrs) in enumerate(wr_by_asset.items()):
        bp = ax4.boxplot(wrs, positions=[i], widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors['test'])
            patch.set_alpha(0.7)
        positions.append(i)
    
    ax4.axhline(33.3, color='gray', ls='--', lw=1, label='Breakeven (33% @ 1:2 RR)')
    ax4.set_xticks(positions)
    ax4.set_xticklabels(list(wr_by_asset.keys()), fontsize=9, rotation=15)
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate Distribution (OOS)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    
    # 5. Max Drawdown
    ax5 = fig.add_subplot(gs[2, 1])
    avg_dd = [np.mean([w['test']['dd'] for w in r['windows']]) for r in valid]
    ax5.bar(assets, avg_dd, color=colors['neg'], alpha=0.7)
    ax5.set_ylabel('Avg Max Drawdown (%)')
    ax5.set_title('Risk: Average Max DD (OOS)', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=15)
    for i, d in enumerate(avg_dd):
        ax5.annotate(f'{d:.1f}%', (i, d+0.5), ha='center', fontsize=9)
    
    # 6. Equity Curves (OOS only)
    ax6 = fig.add_subplot(gs[3, :])
    
    for r in valid:
        combined = [100]
        for curve in r['equity_curves_oos']:
            scale = combined[-1] / 100
            for val in curve[1:]:
                combined.append(val * scale)
        
        color = colors['pos'] if r['avg_oos'] > 0 else colors['neg']
        ax6.plot(combined, label=f"{r['name']} ({r['avg_oos']:+.1f}%)", lw=1.5, alpha=0.8)
    
    ax6.axhline(100, color='gray', ls='-', lw=0.5)
    ax6.set_xlabel('Trade #')
    ax6.set_ylabel('Equity')
    ax6.set_title('Combined OOS Equity Curves (All Windows)', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('CRYPTO VARIANCE PLAY - Walk-Forward (HOURLY DATA)', fontsize=16, fontweight='bold', y=0.98)
    
    chart_path = os.path.join(output_dir, 'walk_forward_hourly.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Chart saved: {chart_path}")
    
    # HTML Report
    html = f"""<!DOCTYPE html>
<html><head><title>Walk-Forward Report (Hourly)</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #00C853; text-align: center; }}
h2 {{ color: #448AFF; }}
.box {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 20px 0; }}
.good {{ border-left: 4px solid #00C853; }}
.warn {{ border-left: 4px solid #FF6D00; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 12px; text-align: center; border: 1px solid #333; }}
th {{ background: #0f3460; }}
.pos {{ color: #00C853; font-weight: bold; }}
.neg {{ color: #FF1744; font-weight: bold; }}
img {{ max-width: 100%; border-radius: 10px; margin: 20px 0; }}
.stat {{ display: inline-block; background: #0f3460; padding: 15px 25px; border-radius: 10px; margin: 10px; text-align: center; }}
.stat-value {{ font-size: 28px; font-weight: bold; }}
.stat-label {{ font-size: 12px; color: #888; }}
</style></head><body>
<h1>🔮 CRYPTO VARIANCE PLAY</h1>
<h2 style="text-align:center; color:#888;">Walk-Forward Blind Test (HOURLY DATA)</h2>

<div class="box good">
<h3>📊 Methodology</h3>
<ul>
<li><strong>Data:</strong> Up to 2 years of HOURLY data (Binance API for crypto)</li>
<li><strong>Windows:</strong> 6 walk-forward periods (90 days train → 30 days test)</li>
<li><strong>Strategy:</strong> Random entries, 2% risk, 1:2 R:R, max 24h hold</li>
<li><strong>Trades per window:</strong> 100 (600 total OOS trades per asset)</li>
</ul>
</div>

<div style="text-align: center;">
"""
    
    # Add summary stats
    viable_count = sum(1 for r in valid if r['avg_oos'] > 0 and r['consistency'] >= 50)
    total_oos = sum(r['total_oos_trades'] for r in valid)
    best = max(valid, key=lambda x: x['avg_oos'])
    
    html += f"""
<div class="stat"><div class="stat-value">{len(valid)}</div><div class="stat-label">Assets Tested</div></div>
<div class="stat"><div class="stat-value">{viable_count}</div><div class="stat-label">Viable Assets</div></div>
<div class="stat"><div class="stat-value">{total_oos:,}</div><div class="stat-label">Total OOS Trades</div></div>
<div class="stat"><div class="stat-value pos">{best['name']}</div><div class="stat-label">Best Performer</div></div>
</div>

<img src="walk_forward_hourly.png" alt="Charts">

<h2>📈 Results</h2>
<table>
<tr><th>Asset</th><th>Data Bars</th><th>OOS Trades</th><th>Avg IS</th><th>Avg OOS</th><th>Consistency</th><th>Avg WR</th><th>Avg DD</th><th>Viable?</th></tr>
"""
    
    # Sort by OOS return
    valid_sorted = sorted(valid, key=lambda x: x['avg_oos'], reverse=True)
    
    for r in valid_sorted:
        viable = r['avg_oos'] > 0 and r['consistency'] >= 50
        avg_wr = np.mean([w['test']['wr'] for w in r['windows']])
        avg_dd = np.mean([w['test']['dd'] for w in r['windows']])
        
        html += f"""<tr>
<td><strong>{r['name']}</strong></td>
<td>{r['total_bars']:,}</td>
<td>{r['total_oos_trades']}</td>
<td class="{'pos' if r['avg_is']>0 else 'neg'}">{r['avg_is']:+.1f}%</td>
<td class="{'pos' if r['avg_oos']>0 else 'neg'}">{r['avg_oos']:+.1f}%</td>
<td>{r['consistency']:.0f}%</td>
<td>{avg_wr:.1f}%</td>
<td>{avg_dd:.1f}%</td>
<td class="{'pos' if viable else 'neg'}">{'✅' if viable else '❌'}</td>
</tr>"""
    
    viable_list = [r['name'] for r in valid_sorted if r['avg_oos'] > 0 and r['consistency'] >= 50]
    
    html += f"""</table>

<div class="box {'good' if viable_list else 'warn'}">
<h3>🎯 Recommendation</h3>
<p><strong>Best performer:</strong> {best['name']} ({best['avg_oos']:+.1f}% avg OOS, {best['consistency']:.0f}% consistency)</p>
<p><strong>Viable assets:</strong> {', '.join(viable_list) if viable_list else 'None passed criteria'}</p>

<h4>For FTMO Crypto CFDs:</h4>
<ul>
<li><strong>Risk:</strong> 0.5-1% per trade (protect daily DD)</li>
<li><strong>R:R:</strong> 1:2 or 1:3</li>
<li><strong>Hold time:</strong> Max 24 hours</li>
<li><strong>Focus:</strong> Momentum entries, not random</li>
</ul>
</div>

<h2>📋 Window Details</h2>
"""
    
    for r in valid_sorted[:3]:  # Top 3 assets
        html += f"""
<h3>{r['name']}</h3>
<table>
<tr><th>Window</th><th>Train Period</th><th>Test Period</th><th>Train Ret</th><th>Test Ret</th><th>Test WR</th></tr>
"""
        for w in r['windows']:
            html += f"""<tr>
<td>W{w['window']}</td>
<td>{w['train_period']}</td>
<td>{w['test_period']}</td>
<td class="{'pos' if w['train']['ret']>0 else 'neg'}">{w['train']['ret']:+.1f}%</td>
<td class="{'pos' if w['test']['ret']>0 else 'neg'}">{w['test']['ret']:+.1f}%</td>
<td>{w['test']['wr']:.1f}%</td>
</tr>"""
        html += "</table>"
    
    html += f"""
<p style="text-align:center; color:#666; margin-top: 40px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Curupira 🌿</p>
</body></html>"""
    
    html_path = os.path.join(output_dir, 'walk_forward_hourly.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"✅ HTML saved: {html_path}")
    
    return chart_path, html_path


def main():
    print("="*70)
    print("WALK-FORWARD BLIND TEST - HOURLY DATA")
    print("="*70)
    print("\nUsing Binance API for crypto (free, up to 2 years hourly)")
    print("Using yfinance for Gold/Forex (limited to 730 days)\n")
    
    # Crypto assets (Binance) - using USDT pairs
    crypto_assets = [
        ("BTCUSDT", "Bitcoin"),
        ("ETHUSDT", "Ethereum"),
        ("SOLUSDT", "Solana"),
        ("DOGEUSDT", "Dogecoin"),
        ("AVAXUSDT", "Avalanche"),
        ("LINKUSDT", "Chainlink"),
        ("ADAUSDT", "Cardano"),
        ("XRPUSDT", "XRP"),
    ]
    
    results = []
    
    # Fetch crypto from Binance (2 years)
    for symbol, name in crypto_assets:
        df = fetch_binance_klines(symbol, interval="1h", days=730)
        if not df.empty:
            res = walk_forward_test(df, name, n_windows=6, 
                                   train_bars=2160, test_bars=720, 
                                   trades_per_window=100)
            results.append(res)
        time.sleep(0.5)  # Rate limit
    
    # Gold from yfinance
    print("\n" + "="*60)
    df_gold = fetch_yfinance_hourly("GC=F")
    if not df_gold.empty:
        res = walk_forward_test(df_gold, "Gold", n_windows=6,
                               train_bars=2160, test_bars=720,
                               trades_per_window=100)
        results.append(res)
    
    valid = [r for r in results if r and r.get('windows')]
    
    if not valid:
        print("\n❌ No valid results")
        return
    
    # Create report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_report(valid, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n{'Asset':<12} {'Bars':<10} {'OOS Trades':<12} {'Avg IS':<10} {'Avg OOS':<10} {'Consistency':<12} {'Viable':<8}")
    print("-"*75)
    
    for r in sorted(valid, key=lambda x: x['avg_oos'], reverse=True):
        viable = "✅" if r['avg_oos'] > 0 and r['consistency'] >= 50 else "❌"
        print(f"{r['name']:<12} {r['total_bars']:<10,} {r['total_oos_trades']:<12} "
              f"{r['avg_is']:>+8.1f}% {r['avg_oos']:>+8.1f}% {r['consistency']:>10.0f}% {viable:>8}")
    
    # Save JSON
    with open(os.path.join(output_dir, 'walk_forward_hourly.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'equity_curves_oos'} for r in valid], f, indent=2, default=str)
    
    print(f"\n✅ Results saved to walk_forward_hourly.json")


if __name__ == "__main__":
    main()
