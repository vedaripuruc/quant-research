"""
Walk-Forward Blind Test - Crypto Variance Play
===============================================
Using DAILY data for 2-year lookback (Yahoo hourly limited to 730 days).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def fetch_data(symbol: str, years: float = 2) -> pd.DataFrame:
    """Fetch daily OHLCV data."""
    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    
    print(f"  Fetching {symbol}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    
    print(f"    {len(df)} daily bars from {df.index[0].date()} to {df.index[-1].date()}")
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
    n_trades: int = 50,
    max_days: int = 5  # Hold max 5 days per trade
) -> Dict:
    """Simulate random entry trades on daily data."""
    if len(df) < max_days * 3:
        return None
    
    results = {"trades": [], "equity_curve": [100.0], "wins": 0, "losses": 0, "timeouts": 0}
    
    valid_idx = list(range(14, len(df) - max_days))  # Need ATR lookback
    if len(valid_idx) < n_trades:
        n_trades = len(valid_idx)
    if n_trades < 10:
        return None
    
    entries = sorted(np.random.choice(valid_idx, n_trades, replace=False))
    equity = 100.0
    
    for idx in entries:
        entry_price = df.iloc[idx]['open']
        
        # ATR from last 14 days
        prev = df.iloc[idx-14:idx]
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
        future = df.iloc[idx:idx+max_days]
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
    
    return {
        "total_trades": total,
        "wins": results["wins"],
        "losses": results["losses"],
        "timeouts": results["timeouts"],
        "win_rate": results["wins"] / total * 100 if total > 0 else 0,
        "final_equity": equity,
        "total_return": equity - 100,
        "max_dd": calculate_max_dd(results["equity_curve"]),
        "equity_curve": results["equity_curve"]
    }


def walk_forward_test(
    symbol: str,
    name: str,
    n_windows: int = 4,
    train_days: int = 180,
    test_days: int = 90,
    trades_per_window: int = 40
) -> Dict:
    """Walk-forward analysis with train/test splits."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {name}")
    print(f"{'='*60}")
    
    df = fetch_data(symbol, years=2)
    
    if df.empty or len(df) < train_days + test_days:
        print(f"  Insufficient data")
        return None
    
    results = {
        "symbol": symbol,
        "name": name,
        "windows": [],
        "in_sample": [],
        "out_of_sample": [],
        "equity_curves_oos": []
    }
    
    total_bars = len(df)
    window_size = train_days + test_days
    step = max((total_bars - window_size) // max(n_windows - 1, 1), test_days)
    
    for i in range(n_windows):
        start = i * step
        if start + window_size > total_bars:
            break
        
        train_df = df.iloc[start:start+train_days]
        test_df = df.iloc[start+train_days:start+window_size]
        
        print(f"\n  Window {i+1}:")
        print(f"    Train: {train_df.index[0].date()} → {train_df.index[-1].date()}")
        print(f"    Test:  {test_df.index[0].date()} → {test_df.index[-1].date()}")
        
        train_res = simulate_trades(train_df, n_trades=trades_per_window)
        test_res = simulate_trades(test_df, n_trades=trades_per_window)
        
        if train_res and test_res:
            results["windows"].append({
                "window": i + 1,
                "train_period": f"{train_df.index[0].date()} → {train_df.index[-1].date()}",
                "test_period": f"{test_df.index[0].date()} → {test_df.index[-1].date()}",
                "train": {"wr": train_res["win_rate"], "ret": train_res["total_return"], "dd": train_res["max_dd"]},
                "test": {"wr": test_res["win_rate"], "ret": test_res["total_return"], "dd": test_res["max_dd"]}
            })
            results["in_sample"].append(train_res["total_return"])
            results["out_of_sample"].append(test_res["total_return"])
            results["equity_curves_oos"].append(test_res["equity_curve"])
            
            print(f"    Train: WR={train_res['win_rate']:.0f}%, Ret={train_res['total_return']:+.1f}%")
            print(f"    Test:  WR={test_res['win_rate']:.0f}%, Ret={test_res['total_return']:+.1f}%")
    
    if results["windows"]:
        results["avg_is"] = np.mean(results["in_sample"])
        results["avg_oos"] = np.mean(results["out_of_sample"])
        results["consistency"] = sum(1 for x in results["out_of_sample"] if x > 0) / len(results["out_of_sample"]) * 100
        results["total_oos_trades"] = sum(1 for w in results["windows"] for _ in range(40))
    
    return results


def create_report(all_results: List[Dict], output_dir: str):
    """Create visual report."""
    valid = [r for r in all_results if r and r.get("windows")]
    
    if not valid:
        print("No valid results to visualize")
        return None, None
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
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
    ax1.set_xticklabels(assets, fontsize=10)
    ax1.set_ylabel('Avg Return (%)')
    ax1.set_title('Walk-Forward: In-Sample vs Out-of-Sample Returns', fontsize=14, fontweight='bold')
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
    for i, c in enumerate(cons):
        ax2.annotate(f'{c:.0f}%', (i, c+2), ha='center', fontsize=10, fontweight='bold')
    
    # 3. Win Rate Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    all_wrs = []
    all_names = []
    for r in valid:
        for w in r['windows']:
            all_wrs.append(w['test']['wr'])
            all_names.append(r['name'])
    
    # Group by asset
    wr_by_asset = {r['name']: [w['test']['wr'] for w in r['windows']] for r in valid}
    positions = []
    for i, (name, wrs) in enumerate(wr_by_asset.items()):
        bp = ax3.boxplot(wrs, positions=[i], widths=0.6, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors['test'])
            patch.set_alpha(0.7)
        positions.append(i)
    
    ax3.axhline(33.3, color='gray', ls='--', lw=1, label='Breakeven (33% @ 1:2 RR)')
    ax3.set_xticks(positions)
    ax3.set_xticklabels(list(wr_by_asset.keys()), fontsize=10)
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate Distribution (OOS)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    
    # 4. Equity Curves (OOS only)
    ax4 = fig.add_subplot(gs[2, :])
    
    for r in valid:
        # Combine all OOS equity curves
        combined = [100]
        for curve in r['equity_curves_oos']:
            # Normalize each curve to continue from previous
            scale = combined[-1] / 100
            for val in curve[1:]:
                combined.append(val * scale)
        
        color = colors['pos'] if r['avg_oos'] > 0 else colors['neg']
        ax4.plot(combined, label=f"{r['name']} ({r['avg_oos']:+.1f}%)", lw=1.5, alpha=0.8)
    
    ax4.axhline(100, color='gray', ls='-', lw=0.5)
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('Equity')
    ax4.set_title('Combined OOS Equity Curves (All Windows)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('CRYPTO VARIANCE PLAY - Walk-Forward Validation', fontsize=16, fontweight='bold', y=0.98)
    
    chart_path = os.path.join(output_dir, 'walk_forward_report.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Chart saved: {chart_path}")
    
    # HTML Report
    html = f"""<!DOCTYPE html>
<html><head><title>Walk-Forward Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
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
</style></head><body>
<h1>🔮 CRYPTO VARIANCE PLAY</h1>
<h2 style="text-align:center; color:#888;">Walk-Forward Blind Test</h2>

<div class="box good">
<h3>📊 Methodology</h3>
<ul>
<li><strong>Data:</strong> 2 years daily OHLCV</li>
<li><strong>Windows:</strong> 180 days train → 90 days test (4 windows)</li>
<li><strong>Strategy:</strong> Random entries, 2% risk, 1:2 R:R, max 5 day hold</li>
<li><strong>Validation:</strong> OOS performance is what matters</li>
</ul></div>

<img src="walk_forward_report.png" alt="Charts">

<h2>📈 Results</h2>
<table>
<tr><th>Asset</th><th>Avg IS</th><th>Avg OOS</th><th>Consistency</th><th>Viable?</th></tr>
"""
    
    for r in valid:
        viable = r['avg_oos'] > 0 and r['consistency'] >= 50
        html += f"""<tr>
<td><strong>{r['name']}</strong></td>
<td class="{'pos' if r['avg_is']>0 else 'neg'}">{r['avg_is']:+.1f}%</td>
<td class="{'pos' if r['avg_oos']>0 else 'neg'}">{r['avg_oos']:+.1f}%</td>
<td>{r['consistency']:.0f}%</td>
<td class="{'pos' if viable else 'neg'}">{'✅' if viable else '❌'}</td>
</tr>"""
    
    # Find best
    best = max(valid, key=lambda x: x['avg_oos'])
    viable_list = [r['name'] for r in valid if r['avg_oos'] > 0 and r['consistency'] >= 50]
    
    html += f"""</table>

<div class="box {'good' if viable_list else 'warn'}">
<h3>🎯 Recommendation</h3>
<p><strong>Best performer:</strong> {best['name']} ({best['avg_oos']:+.1f}% avg OOS)</p>
<p><strong>Viable assets:</strong> {', '.join(viable_list) if viable_list else 'None passed criteria'}</p>
<p><strong>For FTMO:</strong></p>
<ul>
<li>Risk: 0.5-1% per trade</li>
<li>R:R: 1:2 or 1:3</li>
<li>Focus on momentum, not random entries</li>
</ul></div>

<p style="text-align:center; color:#666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Curupira 🌿</p>
</body></html>"""
    
    html_path = os.path.join(output_dir, 'walk_forward_report.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"✅ HTML saved: {html_path}")
    
    return chart_path, html_path


def main():
    print("="*70)
    print("WALK-FORWARD BLIND TEST - CRYPTO VARIANCE")
    print("="*70)
    print("\nUsing DAILY data for 2-year lookback\n")
    
    assets = [
        ("BTC-USD", "Bitcoin"),
        ("ETH-USD", "Ethereum"),
        ("SOL-USD", "Solana"),
        ("DOGE-USD", "Dogecoin"),
        ("AVAX-USD", "Avalanche"),
        ("LINK-USD", "Chainlink"),
        ("GC=F", "Gold"),
        ("EURUSD=X", "EUR/USD"),
    ]
    
    results = []
    for sym, name in assets:
        res = walk_forward_test(sym, name, n_windows=4, train_days=180, test_days=90, trades_per_window=40)
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
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Asset':<12} {'Avg IS':<10} {'Avg OOS':<10} {'Consistency':<12} {'Viable':<8}")
    print("-"*55)
    
    for r in valid:
        viable = "✅" if r['avg_oos'] > 0 and r['consistency'] >= 50 else "❌"
        print(f"{r['name']:<12} {r['avg_is']:>+8.1f}% {r['avg_oos']:>+8.1f}% {r['consistency']:>10.0f}% {viable:>8}")
    
    # Save JSON
    with open(os.path.join(output_dir, 'walk_forward_results.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'equity_curves_oos'} for r in valid], f, indent=2, default=str)
    
    print(f"\n✅ Results saved to walk_forward_results.json")


if __name__ == "__main__":
    main()
