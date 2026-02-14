#!/usr/bin/env python3
"""
Signal chart generator — focused matplotlib JPEGs for each active signal.
Big readable charts with entry/SL/TP levels, shaded risk/reward zones.
"""

import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime


DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style='nightclouds',
    marketcolors=mpf.make_marketcolors(
        up='#00ff88', down='#ff4444',
        wick={'up': '#00ff8888', 'down': '#ff444488'},
        edge={'up': '#00ff88', 'down': '#ff4444'},
        volume='#333366',
    ),
    facecolor='#0a0a0f',
    edgecolor='#1a1a2e',
    figcolor='#0a0a0f',
    gridcolor='#151525',
    gridstyle='--',
    gridaxis='both',
    y_on_right=True,
    rc={
        'font.size': 12,
        'axes.labelcolor': '#888',
        'xtick.color': '#666',
        'ytick.color': '#666',
    },
)


def generate_signal_chart(
    symbol: str,
    asset_name: str,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    signal_type: str,
    details: str,
    output_dir: Path,
    period: str = "14d",
    interval: str = "1h",
) -> Path | None:
    """Generate a focused JPEG chart for a single signal."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception as e:
        print(f"  Chart error fetching {symbol}: {e}")
        return None

    if df is None or df.empty or len(df) < 10:
        print(f"  Chart: insufficient data for {symbol}")
        return None

    df.index.name = 'Date'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Focus: last 80 bars (or all if less)
    focus_bars = min(80, len(df))
    df = df.tail(focus_bars)

    # Determine y-axis range to include all levels
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    all_levels = [entry, sl, tp, price_min, price_max]
    y_min = min(all_levels)
    y_max = max(all_levels)
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    # Type labels
    type_labels = {
        'jump_fade': 'Jump Fade', 'jump_trend': 'Jump Trend',
        'hurst_trend': 'Hurst Trend', 'hurst_fade': 'Hurst Fade',
    }
    type_str = type_labels.get(signal_type, signal_type)

    # Price format
    if symbol == 'GC=F':
        fmt_str = ',.2f'
    elif entry > 1:
        fmt_str = '.4f'
    else:
        fmt_str = '.6f'

    title = f"  {direction} {asset_name}  |  {type_str}  |  R:R 1:2  "

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = asset_name.lower().replace(' ', '_')
    out_path = output_dir / f"signal_{safe_name}.jpg"

    # Draw the chart
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=DARK_STYLE,
        title=title,
        volume=False,
        figsize=(16, 8),
        ylim=(y_min, y_max),
        returnfig=True,
        tight_layout=True,
    )

    ax = axes[0]
    n = len(df)

    # ── Shaded zones ──
    # Risk zone (entry → SL) = red
    # Reward zone (entry → TP) = green
    if direction == 'LONG':
        ax.axhspan(sl, entry, alpha=0.08, color='#ff4444')   # risk below
        ax.axhspan(entry, tp, alpha=0.06, color='#00ff88')   # reward above
    else:
        ax.axhspan(entry, sl, alpha=0.08, color='#ff4444')   # risk above
        ax.axhspan(tp, entry, alpha=0.06, color='#00ff88')   # reward below

    # ── Horizontal level lines spanning full chart ──
    ax.axhline(y=entry, color='#4488ff', linewidth=2.0, linestyle='-', alpha=0.9)
    ax.axhline(y=sl, color='#ff4444', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(y=tp, color='#00ff88', linewidth=1.5, linestyle='--', alpha=0.8)

    # ── Labels on the LEFT side of chart ──
    label_x = 1  # near left edge

    ax.text(label_x, entry, f'  ENTRY ${entry:{fmt_str}}  ',
            fontsize=11, fontweight='bold', color='#4488ff',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#4488ff33', edgecolor='#4488ff88', linewidth=1.5))

    ax.text(label_x, sl, f'  SL ${sl:{fmt_str}}  ',
            fontsize=10, fontweight='bold', color='#ff4444',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ff444433', edgecolor='#ff444488', linewidth=1.5))

    ax.text(label_x, tp, f'  TP ${tp:{fmt_str}}  ',
            fontsize=10, fontweight='bold', color='#00ff88',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#00ff8833', edgecolor='#00ff8888', linewidth=1.5))

    # ── R:R and distance annotations on RIGHT side ──
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    right_x = n - 2

    ax.text(right_x, sl, f'  {sl_dist:{fmt_str}} risk  ',
            fontsize=9, color='#ff6666', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff444422', edgecolor='none'))

    ax.text(right_x, tp, f'  {tp_dist:{fmt_str}} reward  ',
            fontsize=9, color='#66ff99', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#00ff8822', edgecolor='none'))

    # ── Details at bottom ──
    fig.text(0.5, 0.01, details, ha='center', fontsize=10, color='#777', fontstyle='italic')

    # ── Direction arrow marker on the last candle ──
    last_close = float(df['Close'].iloc[-1])
    arrow_color = '#00ff88' if direction == 'LONG' else '#ff4444'
    arrow_dir = 0.3 if direction == 'LONG' else -0.3
    # Arrow: xytext=start, xy=arrowhead. LONG arrow points UP, SHORT points DOWN.
    ax.annotate('',
                xy=(n - 1, entry + (y_max - y_min) * arrow_dir),  # arrowhead
                xytext=(n - 1, entry),  # start
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close(fig)

    print(f"  Chart: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
    return out_path


def generate_all_signal_charts(signals: list[dict], output_dir: Path) -> dict[str, str]:
    """Generate charts for all signals. Returns {asset_name: filename}."""
    results = {}
    for sig in signals:
        symbol = sig.get('symbol', '')
        asset = sig.get('asset', '')

        period = '30d' if symbol == 'GC=F' else '14d'

        path = generate_signal_chart(
            symbol=symbol,
            asset_name=asset,
            direction=sig.get('direction', ''),
            entry=sig.get('entry', 0),
            sl=sig.get('stop_loss', 0),
            tp=sig.get('take_profit', 0),
            signal_type=sig.get('signal_type', ''),
            details=sig.get('details', ''),
            output_dir=output_dir,
            period=period,
            interval='1h',
        )
        if path:
            results[asset] = path.name

    return results


if __name__ == '__main__':
    test_signals = [
        {
            'asset': 'Gold', 'symbol': 'GC=F', 'direction': 'SHORT',
            'entry': 4950.0, 'stop_loss': 5202.66, 'take_profit': 4444.69,
            'signal_type': 'jump_fade',
            'details': 'Jump ratio: 0.560, fading +1.4% move',
        },
        {
            'asset': 'XRP', 'symbol': 'XRP-USD', 'direction': 'SHORT',
            'entry': 1.3613, 'stop_loss': 1.5632, 'take_profit': 0.9576,
            'signal_type': 'hurst_trend',
            'details': 'Hurst: 0.637, EMA slope: -0.022 (trending down)',
        },
    ]
    out = Path('data/signals')
    results = generate_all_signal_charts(test_signals, out)
    print(f"\nGenerated {len(results)} charts: {results}")
