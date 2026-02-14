#!/usr/bin/env python3
"""
Trade Visualization Dashboard
-----------------------------
Interactive charts to visually verify trades.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import STRATEGIES, williams_r_indicator, rsi_indicator


def create_trade_chart(df: pd.DataFrame, trades_df: pd.DataFrame, 
                       symbol: str, strategy: str, metrics: dict) -> go.Figure:
    """Create an interactive candlestick chart with trade markers."""
    
    # Create figure with secondary y-axis for indicators
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - {strategy.upper()}', 'Indicator')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add trades if any
    if not trades_df.empty:
        completed = trades_df[trades_df['exit_price'].notna()]
        
        # Long entries (green triangles pointing up)
        longs = completed[completed['direction'] == 'long']
        if len(longs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=longs['entry_time'],
                    y=longs['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='#00c853',
                        line=dict(width=2, color='white')
                    ),
                    name='Long Entry',
                    hovertemplate='<b>LONG ENTRY</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Short entries (red triangles pointing down)
        shorts = completed[completed['direction'] == 'short']
        if len(shorts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=shorts['entry_time'],
                    y=shorts['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#ff1744',
                        line=dict(width=2, color='white')
                    ),
                    name='Short Entry',
                    hovertemplate='<b>SHORT ENTRY</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Exits - wins (blue circles)
        wins = completed[completed['result'] == 'win']
        if len(wins) > 0:
            fig.add_trace(
                go.Scatter(
                    x=wins['exit_time'],
                    y=wins['exit_price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='#2196f3',
                        line=dict(width=2, color='white')
                    ),
                    name='Win Exit',
                    hovertemplate='<b>WIN EXIT</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Exits - losses (orange X)
        losses = completed[completed['result'] == 'loss']
        if len(losses) > 0:
            fig.add_trace(
                go.Scatter(
                    x=losses['exit_time'],
                    y=losses['exit_price'],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='#ff9800',
                        line=dict(width=3)
                    ),
                    name='Loss Exit',
                    hovertemplate='<b>LOSS EXIT</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Price: $%{y:.2f}<br>' +
                                  '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Draw lines connecting entry to exit
        for _, trade in completed.iterrows():
            color = '#00c853' if trade['result'] == 'win' else '#ff9800'
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Add SL/TP lines
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['stop_loss'], trade['stop_loss']],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=False,
                    opacity=0.5,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['take_profit'], trade['take_profit']],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    showlegend=False,
                    opacity=0.5,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    # Add indicator based on strategy
    if strategy == 'williams_r':
        wr = williams_r_indicator(df)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=wr, name='Williams %R', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=-80, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)
    
    elif strategy == 'rsi_divergence':
        rsi = rsi_indicator(df['Close'])
        fig.add_trace(
            go.Scatter(x=df['Date'], y=rsi, name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    elif strategy == 'ma_crossover':
        fast_ma = df['Close'].rolling(10).mean()
        slow_ma = df['Close'].rolling(30).mean()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=fast_ma, name='Fast MA (10)', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=slow_ma, name='Slow MA (30)', line=dict(color='orange')),
            row=1, col=1
        )
        # Volume in subplot
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='gray', opacity=0.5),
            row=2, col=1
        )
    
    elif strategy == 'breakout':
        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        fig.add_trace(
            go.Scatter(x=df['Date'], y=highest, name='20-bar High', 
                       line=dict(color='green', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=lowest, name='20-bar Low',
                       line=dict(color='red', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='gray', opacity=0.5),
            row=2, col=1
        )
    
    else:
        # Default: just show volume
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='gray', opacity=0.5),
            row=2, col=1
        )
    
    # Update layout
    beat_bnh = "✓" if metrics['strategy_return'] > metrics['buy_hold_return'] else "✗"
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} - {strategy.upper()}</b><br>" +
                 f"<sup>Trades: {metrics['total_trades']} | " +
                 f"WR: {metrics['win_rate']:.1f}% | " +
                 f"Return: {metrics['strategy_return']:.2f}% vs B&H: {metrics['buy_hold_return']:.2f}% {beat_bnh} | " +
                 f"Max DD: {metrics['max_drawdown']:.2f}%</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_rangeslider_visible=False,
        height=700,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    fig.update_xaxes(type='category', tickangle=45, row=2, col=1)
    
    return fig


def generate_dashboard(symbols: list = None, strategies: list = None, 
                       output_dir: str = 'charts'):
    """Generate HTML dashboard with all charts."""
    
    symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'META', 'AMZN']
    strategies = strategies or list(STRATEGIES.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.001,
        position_size=1.0,
        compound=True
    )
    
    all_charts = []
    
    print("Generating charts...")
    
    for strategy_name in strategies:
        for symbol in symbols:
            print(f"  {symbol} - {strategy_name}...", end=" ", flush=True)
            
            try:
                df = fetch_data(symbol, '1h', 59)
                if df.empty or len(df) < 30:
                    print("SKIP (no data)")
                    continue
                
                engine = BacktestEngine(config)
                trades_df = engine.run(df, STRATEGIES[strategy_name])
                metrics = calculate_metrics(trades_df, df, config)
                
                fig = create_trade_chart(df, trades_df, symbol, strategy_name, metrics)
                
                # Save individual chart
                filename = f"{symbol}_{strategy_name}.html"
                filepath = os.path.join(output_dir, filename)
                fig.write_html(filepath, include_plotlyjs='cdn')
                
                all_charts.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'filename': filename,
                    'metrics': metrics
                })
                
                print(f"OK ({metrics['total_trades']}t)")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Generate index page
    index_html = generate_index(all_charts, output_dir)
    
    print(f"\n✓ Generated {len(all_charts)} charts")
    print(f"✓ Open: file://{os.path.abspath(output_dir)}/index.html")
    
    return output_dir


def generate_index(charts: list, output_dir: str) -> str:
    """Generate index.html with links to all charts."""
    
    # Group by strategy
    by_strategy = {}
    for c in charts:
        strat = c['strategy']
        if strat not in by_strategy:
            by_strategy[strat] = []
        by_strategy[strat].append(c)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Curupira Backtests - Visual Audit</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 { 
            text-align: center; 
            margin-bottom: 10px;
            color: #00d4aa;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        .strategy-section {
            margin-bottom: 40px;
        }
        .strategy-title {
            font-size: 1.5em;
            color: #ff6b6b;
            margin-bottom: 15px;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .chart-card {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chart-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,212,170,0.2);
        }
        .chart-card a {
            color: #00d4aa;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.2em;
        }
        .metrics {
            margin-top: 10px;
            font-size: 0.9em;
            color: #aaa;
        }
        .metrics span {
            display: inline-block;
            margin-right: 15px;
        }
        .win { color: #00c853; }
        .loss { color: #ff5252; }
        .neutral { color: #888; }
        .beat { color: #00d4aa; font-weight: bold; }
        .miss { color: #ff6b6b; }
        .legend {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
        }
        .legend-marker {
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <h1>🌿 Curupira Backtests - Visual Audit</h1>
    <p class="subtitle">Click any chart to inspect trades. Zoom, hover, verify.</p>
    
    <div class="legend">
        <b>Legend:</b>
        <span class="legend-item"><span class="legend-marker" style="background:#00c853;border-radius:50%;"></span> Long Entry (▲)</span>
        <span class="legend-item"><span class="legend-marker" style="background:#ff1744;border-radius:50%;"></span> Short Entry (▼)</span>
        <span class="legend-item"><span class="legend-marker" style="background:#2196f3;border-radius:50%;"></span> Win Exit (●)</span>
        <span class="legend-item"><span class="legend-marker" style="background:#ff9800;border-radius:50%;"></span> Loss Exit (✗)</span>
        <span class="legend-item">--- TP/SL levels</span>
    </div>
"""
    
    for strategy, charts_list in by_strategy.items():
        # Calculate strategy totals
        total_trades = sum(c['metrics']['total_trades'] for c in charts_list)
        avg_wr = np.mean([c['metrics']['win_rate'] for c in charts_list])
        avg_ret = np.mean([c['metrics']['strategy_return'] for c in charts_list])
        beat_count = sum(1 for c in charts_list 
                        if c['metrics']['strategy_return'] > c['metrics']['buy_hold_return'])
        
        html += f"""
    <div class="strategy-section">
        <h2 class="strategy-title">{strategy.upper()} 
            <span style="font-size:0.6em;color:#888;">
                ({total_trades} trades | {avg_wr:.1f}% WR | {avg_ret:.2f}% avg | {beat_count}/{len(charts_list)} beat B&H)
            </span>
        </h2>
        <div class="chart-grid">
"""
        
        for c in sorted(charts_list, key=lambda x: -x['metrics']['strategy_return']):
            m = c['metrics']
            ret_class = 'win' if m['strategy_return'] > 0 else 'loss' if m['strategy_return'] < 0 else 'neutral'
            beat_class = 'beat' if m['strategy_return'] > m['buy_hold_return'] else 'miss'
            
            html += f"""
            <div class="chart-card">
                <a href="{c['filename']}">{c['symbol']}</a>
                <div class="metrics">
                    <span>Trades: {m['total_trades']}</span>
                    <span>WR: {m['win_rate']:.1f}%</span>
                    <span class="{ret_class}">Ret: {m['strategy_return']:.2f}%</span>
                    <span class="{beat_class}">vs B&H: {m['buy_hold_return']:.2f}%</span>
                </div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += f"""
    <p style="text-align:center;color:#555;margin-top:40px;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Config: 0.05% slippage, 0.1% commission, compounded
    </p>
</body>
</html>
"""
    
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html)
    
    return index_path


if __name__ == "__main__":
    import sys
    
    # Quick single chart for testing
    if len(sys.argv) > 2:
        symbol = sys.argv[1]
        strategy = sys.argv[2]
        generate_dashboard([symbol], [strategy])
    else:
        generate_dashboard()
