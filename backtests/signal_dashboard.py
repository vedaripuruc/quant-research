#!/usr/bin/env python3
"""
Signal Dashboard Generator
--------------------------
Creates an HTML dashboard with charts for each signal.
You can visually review before taking a trade.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIG
# ============================================================================

STATE_FILE = Path(__file__).parent / 'signal_state.json'
DASHBOARD_FILE = Path(__file__).parent / 'charts' / 'signals_dashboard.html'

# How many hours of data to show on chart
CHART_HOURS = 72  # 3 days

# ============================================================================
# DATA
# ============================================================================

def fetch_recent_data(symbol, hours=72):
    """Fetch recent hourly data for charting"""
    try:
        days = max(7, hours // 24 + 2)  # Need buffer for yfinance
        end = datetime.now()
        start = end - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'), 
                           end=end.strftime('%Y-%m-%d'),
                           interval='1h')
        
        if df.empty:
            return None
        
        # Keep only last N hours
        df = df.tail(hours)
        df.reset_index(inplace=True)
        
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df):
    """Calculate strategy indicators"""
    # Session range (last 4 bars before current)
    if len(df) >= 5:
        session = df.iloc[-5:-1]
        df.loc[:, 'session_high'] = session['High'].max()
        df.loc[:, 'session_low'] = session['Low'].min()
    
    # Donchian channel (20 period)
    df['donchian_high'] = df['High'].rolling(20).max()
    df['donchian_low'] = df['Low'].rolling(20).min()
    
    # ATR for context
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df

# ============================================================================
# CHART GENERATION
# ============================================================================

def create_signal_chart(signal, df):
    """Create a detailed chart for a signal"""
    
    symbol = signal['symbol']
    signal_type = signal['type']
    strategy = signal['strategy']
    entry = signal['entry']
    sl = signal['stop_loss']
    tp = signal['take_profit']
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=(f'{symbol} - {signal_type} Signal ({strategy})', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )
    
    # Session range (if session_breakout)
    if strategy == 'session_breakout' and 'session_high' in df.columns:
        session_high = df['session_high'].iloc[-1]
        session_low = df['session_low'].iloc[-1]
        
        # Session range box
        fig.add_hrect(
            y0=session_low, y1=session_high,
            fillcolor="rgba(255, 235, 59, 0.2)",
            line=dict(color="rgba(255, 235, 59, 0.5)", width=1),
            annotation_text="Session Range",
            annotation_position="top left",
            row=1, col=1
        )
    
    # Donchian channel (if breakout)
    if strategy == 'breakout':
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['donchian_high'],
                mode='lines',
                line=dict(color='rgba(76, 175, 80, 0.5)', width=1, dash='dot'),
                name='20-bar High'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'], y=df['donchian_low'],
                mode='lines',
                line=dict(color='rgba(244, 67, 54, 0.5)', width=1, dash='dot'),
                name='20-bar Low'
            ),
            row=1, col=1
        )
    
    # Entry level
    fig.add_hline(
        y=entry,
        line=dict(color='white', width=2),
        annotation_text=f"Entry: {entry:.5f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Stop Loss
    fig.add_hline(
        y=sl,
        line=dict(color='#ff1744', width=2, dash='dash'),
        annotation_text=f"SL: {sl:.5f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Take Profit
    fig.add_hline(
        y=tp,
        line=dict(color='#00e676', width=2, dash='dash'),
        annotation_text=f"TP: {tp:.5f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Current price marker
    current_price = df['Close'].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[df['Date'].iloc[-1]],
            y=[current_price],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='diamond'),
            name=f'Current: {current_price:.5f}'
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['#26a69a' if c >= o else '#ef5350' 
              for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    # Calculate risk/reward
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>🎯 {signal_type} {symbol}</b><br>"
                 f"<sup>Strategy: {strategy} | "
                 f"Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | "
                 f"R:R = 1:{rr:.1f}</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        height=600,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def generate_dashboard():
    """Generate full dashboard with all signals"""
    
    # Load state
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except:
        print("No signal state found. Run signal_monitor.py --once first.")
        return
    
    signals = state.get('signals', [])
    scan_time = state.get('scan_time', 'Unknown')
    
    # Start HTML
    html_parts = [f'''<!DOCTYPE html>
<html>
<head>
    <title>Trading Signals Dashboard</title>
    <meta http-equiv="refresh" content="900"> <!-- Refresh every 15 min -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
        }}
        .header h1 {{ color: #00d4aa; margin-bottom: 10px; }}
        .header .time {{ color: #888; }}
        .header .count {{ font-size: 1.5em; color: #ff6b6b; margin-top: 10px; }}
        .signal-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }}
        .signal-type {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .signal-type.long {{ color: #00c853; }}
        .signal-type.short {{ color: #ff5252; }}
        .signal-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        .meta-item {{
            background: #0f0f23;
            padding: 10px;
            border-radius: 5px;
        }}
        .meta-label {{ color: #888; font-size: 0.8em; }}
        .meta-value {{ font-size: 1.1em; font-weight: bold; }}
        .meta-value.entry {{ color: #fff; }}
        .meta-value.sl {{ color: #ff5252; }}
        .meta-value.tp {{ color: #00c853; }}
        .chart-container {{
            width: 100%;
            height: 600px;
        }}
        .no-signals {{
            text-align: center;
            padding: 50px;
            color: #888;
            font-size: 1.5em;
        }}
        .data-source {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Trading Signals Dashboard</h1>
        <div class="time">Last scan: {scan_time}</div>
        <div class="count">{len(signals)} Active Signal{"s" if len(signals) != 1 else ""}</div>
    </div>
''']
    
    if not signals:
        html_parts.append('''
    <div class="no-signals">
        😴 No active signals at this time.<br>
        <small>Dashboard auto-refreshes every 15 minutes.</small>
    </div>
''')
    else:
        # Deduplicate signals (keep one per symbol, prefer session_breakout)
        seen = {}
        for s in signals:
            key = s['symbol']
            if key not in seen or s['strategy'] == 'session_breakout':
                seen[key] = s
        
        unique_signals = list(seen.values())
        
        for i, signal in enumerate(unique_signals):
            symbol = signal['symbol']
            signal_type = signal['type']
            strategy = signal['strategy']
            entry = signal['entry']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            
            # Calculate R:R
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            
            # Fetch data and create chart
            df = fetch_recent_data(symbol, CHART_HOURS)
            
            if df is not None and len(df) > 0:
                fig = create_signal_chart(signal, df)
                chart_json = fig.to_json()
            else:
                chart_json = None
            
            type_class = 'long' if signal_type == 'LONG' else 'short'
            
            html_parts.append(f'''
    <div class="signal-card">
        <div class="signal-header">
            <span class="signal-type {type_class}">{signal_type} {symbol.replace("=X", "")}</span>
            <span style="color: #888;">Strategy: {strategy}</span>
        </div>
        <div class="signal-meta">
            <div class="meta-item">
                <div class="meta-label">Entry</div>
                <div class="meta-value entry">{entry:.5f}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Stop Loss</div>
                <div class="meta-value sl">{sl:.5f}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Take Profit</div>
                <div class="meta-value tp">{tp:.5f}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Risk:Reward</div>
                <div class="meta-value">1:{rr:.1f}</div>
            </div>
        </div>
''')
            
            if chart_json:
                html_parts.append(f'''
        <div class="chart-container" id="chart_{i}"></div>
        <script>
            var data_{i} = {chart_json};
            Plotly.newPlot('chart_{i}', data_{i}.data, data_{i}.layout, {{responsive: true}});
        </script>
''')
            else:
                html_parts.append('''
        <div style="text-align:center;padding:50px;color:#888;">
            Chart unavailable - could not fetch data
        </div>
''')
            
            html_parts.append('    </div>\n')
    
    # Footer
    html_parts.append('''
    <div class="data-source">
        <strong>Data Source:</strong> Yahoo Finance (may be delayed 15-20min for forex)<br>
        <strong>Strategies:</strong> Session Breakout (4-bar range) | Donchian Breakout (20-bar high/low)<br>
        <strong>Auto-refresh:</strong> Every 15 minutes
    </div>
</body>
</html>
''')
    
    # Write dashboard
    os.makedirs(DASHBOARD_FILE.parent, exist_ok=True)
    with open(DASHBOARD_FILE, 'w') as f:
        f.write(''.join(html_parts))
    
    print(f"Dashboard generated: {DASHBOARD_FILE}")
    print(f"Signals: {len(signals)}")
    print(f"Open: http://localhost:8888/signals_dashboard.html")


if __name__ == '__main__':
    generate_dashboard()
