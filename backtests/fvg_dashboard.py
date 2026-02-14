#!/usr/bin/env python3
"""
FVG Visualization Dashboard
---------------------------
Dedicated frontend for visualizing Fair Value Gap research results.
Serves on port 8889 (8888 is main dashboard).

Usage:
    python fvg_dashboard.py
    # Then open http://localhost:8889/fvg
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIG
# ============================================================================

app = Flask(__name__, template_folder='templates')

BASE_DIR = Path(__file__).parent
FVG_RESULTS_FILE = BASE_DIR / 'fvg_research_results.json'
BACKTEST_RESULTS_FILE = BASE_DIR / 'backtest_results_v2.json'

# FVG Variations with colors
FVG_VARIATIONS = {
    'fvg_magnet': {'name': 'FVG Magnet', 'color': '#58a6ff', 'desc': 'Price attracted to gap midpoint'},
    'fvg_wall': {'name': 'FVG Wall', 'color': '#f85149', 'desc': 'Gap acts as support/resistance'},
    'fvg_fill_reversal': {'name': 'FVG Fill Reversal', 'color': '#3fb950', 'desc': 'Reversal after gap fill'},
    'fvg_edge': {'name': 'FVG Edge', 'color': '#d29922', 'desc': 'Trade from gap edge'},
    'fvg': {'name': 'FVG Basic', 'color': '#8b949e', 'desc': 'Original FVG strategy'},
}

FOREX_PAIRS = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'EURJPY=X', 'AUDUSD=X', 'USDCAD=X']

# ============================================================================
# DATA LOADING
# ============================================================================

def load_fvg_results():
    """Load FVG research results or fall back to backtest data"""
    
    # Try dedicated FVG results first
    if FVG_RESULTS_FILE.exists():
        with open(FVG_RESULTS_FILE, 'r') as f:
            return json.load(f)
    
    # Fall back to extracting FVG data from main backtest results
    if BACKTEST_RESULTS_FILE.exists():
        with open(BACKTEST_RESULTS_FILE, 'r') as f:
            data = json.load(f)
        
        # Filter for FVG-related strategies
        fvg_results = []
        for result in data.get('results', []):
            strategy = result.get('strategy', '')
            if 'fvg' in strategy.lower():
                fvg_results.append(result)
        
        return {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'source': 'backtest_results_v2.json',
            'variations': list(FVG_VARIATIONS.keys()),
            'results': fvg_results
        }
    
    # Return mock data for development
    return generate_mock_fvg_data()


def generate_mock_fvg_data():
    """Generate mock FVG data for dashboard development"""
    results = []
    
    for pair in FOREX_PAIRS[:4]:
        for var_key, var_info in FVG_VARIATIONS.items():
            results.append({
                'symbol': pair,
                'strategy': var_key,
                'variation': var_info['name'],
                'category': 'Forex',
                'status': 'pending',
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'strategy_return': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'expectancy': 0,
                'sharpe': 0,
                'equity_curve': [],
                'trades': []
            })
    
    return {
        'timestamp': datetime.now().isoformat(),
        'source': 'mock_data',
        'note': 'FVG research not yet completed. Run fvg_research.py first.',
        'variations': list(FVG_VARIATIONS.keys()),
        'results': results
    }


def fetch_ohlc_data(symbol, days=30):
    """Fetch OHLC data for charting"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'),
                           interval='1h')
        
        if df.empty:
            return None
        
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def detect_fvg_zones(df):
    """Detect Fair Value Gaps in OHLC data"""
    fvg_zones = []
    
    for i in range(2, len(df)):
        # Bullish FVG: bar[i-2].high < bar[i].low (gap up)
        if df.iloc[i-2]['High'] < df.iloc[i]['Low']:
            fvg_zones.append({
                'type': 'bullish',
                'start_idx': i - 2,
                'end_idx': i,
                'date': df.iloc[i]['Date'],
                'top': df.iloc[i]['Low'],
                'bottom': df.iloc[i-2]['High'],
                'midpoint': (df.iloc[i]['Low'] + df.iloc[i-2]['High']) / 2,
            })
        
        # Bearish FVG: bar[i-2].low > bar[i].high (gap down)
        if df.iloc[i-2]['Low'] > df.iloc[i]['High']:
            fvg_zones.append({
                'type': 'bearish',
                'start_idx': i - 2,
                'end_idx': i,
                'date': df.iloc[i]['Date'],
                'top': df.iloc[i-2]['Low'],
                'bottom': df.iloc[i]['High'],
                'midpoint': (df.iloc[i-2]['Low'] + df.iloc[i]['High']) / 2,
            })
    
    return fvg_zones


# ============================================================================
# CHART GENERATION
# ============================================================================

def create_fvg_chart(symbol, variation=None):
    """Create interactive FVG chart with Plotly"""
    df = fetch_ohlc_data(symbol)
    
    if df is None or df.empty:
        return None
    
    # Detect FVG zones
    fvg_zones = detect_fvg_zones(df)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - FVG Analysis', 'Volume')
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
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149'
        ),
        row=1, col=1
    )
    
    # Add FVG zones
    for zone in fvg_zones:
        zone_color = 'rgba(63, 185, 80, 0.2)' if zone['type'] == 'bullish' else 'rgba(248, 81, 73, 0.2)'
        border_color = '#3fb950' if zone['type'] == 'bullish' else '#f85149'
        
        # Start and extend the zone for visibility
        start_date = df.iloc[zone['start_idx']]['Date']
        end_idx = min(zone['end_idx'] + 20, len(df) - 1)  # Extend zone forward
        end_date = df.iloc[end_idx]['Date']
        
        # Zone rectangle
        fig.add_shape(
            type='rect',
            x0=start_date, x1=end_date,
            y0=zone['bottom'], y1=zone['top'],
            fillcolor=zone_color,
            line=dict(color=border_color, width=1),
            row=1, col=1
        )
        
        # Midpoint line (dashed)
        fig.add_shape(
            type='line',
            x0=start_date, x1=end_date,
            y0=zone['midpoint'], y1=zone['midpoint'],
            line=dict(color=border_color, width=1, dash='dash'),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['#3fb950' if row['Close'] >= row['Open'] else '#f85149' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume', showlegend=False),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        font=dict(color='#c9d1d9')
    )
    
    fig.update_xaxes(gridcolor='#30363d', showgrid=True)
    fig.update_yaxes(gridcolor='#30363d', showgrid=True)
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_equity_chart(results, pair_filter=None):
    """Create equity curve comparison chart"""
    fig = go.Figure()
    
    # Group results by variation
    variation_data = {}
    
    for result in results:
        var_key = result.get('strategy', 'unknown')
        symbol = result.get('symbol', '')
        
        # Apply pair filter if specified
        if pair_filter and symbol != pair_filter:
            continue
        
        if var_key not in variation_data:
            variation_data[var_key] = {
                'equity': result.get('equity_curve', []),
                'color': FVG_VARIATIONS.get(var_key, {}).get('color', '#8b949e')
            }
    
    # Add equity curves
    for var_key, data in variation_data.items():
        equity = data['equity']
        if equity:
            fig.add_trace(go.Scatter(
                y=equity,
                mode='lines',
                name=FVG_VARIATIONS.get(var_key, {}).get('name', var_key),
                line=dict(color=data['color'], width=2)
            ))
        else:
            # Mock equity curve if no data
            initial = 10000
            returns = np.random.normal(0.0001, 0.005, 100)
            equity = [initial]
            for r in returns:
                equity.append(equity[-1] * (1 + r))
            
            fig.add_trace(go.Scatter(
                y=equity,
                mode='lines',
                name=f"{FVG_VARIATIONS.get(var_key, {}).get('name', var_key)} (simulated)",
                line=dict(color=data['color'], width=2, dash='dot')
            ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        title='Equity Curve Comparison',
        xaxis_title='Trade #',
        yaxis_title='Portfolio Value ($)',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        font=dict(color='#c9d1d9')
    )
    
    fig.update_xaxes(gridcolor='#30363d')
    fig.update_yaxes(gridcolor='#30363d')
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_drawdown_chart(results, pair_filter=None):
    """Create drawdown comparison chart"""
    fig = go.Figure()
    
    for result in results:
        var_key = result.get('strategy', 'unknown')
        symbol = result.get('symbol', '')
        
        if pair_filter and symbol != pair_filter:
            continue
        
        equity = result.get('equity_curve', [])
        if not equity:
            # Mock data
            initial = 10000
            returns = np.random.normal(0.0001, 0.005, 100)
            equity = [initial]
            for r in returns:
                equity.append(equity[-1] * (1 + r))
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = [(e - p) / p * 100 for e, p in zip(equity, peak)]
        
        color = FVG_VARIATIONS.get(var_key, {}).get('color', '#8b949e')
        
        fig.add_trace(go.Scatter(
            y=drawdown,
            mode='lines',
            name=FVG_VARIATIONS.get(var_key, {}).get('name', var_key),
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in color else f'{color}1a'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        title='Drawdown Comparison',
        xaxis_title='Trade #',
        yaxis_title='Drawdown (%)',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(color='#c9d1d9')
    )
    
    fig.update_xaxes(gridcolor='#30363d')
    fig.update_yaxes(gridcolor='#30363d')
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/fvg')
def fvg_overview():
    """FVG Research Overview page"""
    data = load_fvg_results()
    
    # Get unique pairs
    pairs = list(set(r.get('symbol', '') for r in data.get('results', [])))
    pairs.sort()
    
    # Get filter from query params
    pair_filter = request.args.get('pair', '')
    sort_by = request.args.get('sort', 'strategy_return')
    sort_dir = request.args.get('dir', 'desc')
    
    # Filter results
    results = data.get('results', [])
    if pair_filter:
        results = [r for r in results if r.get('symbol') == pair_filter]
    
    # Sort results
    reverse = sort_dir == 'desc'
    try:
        results = sorted(results, key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)
    except:
        pass
    
    return render_template(
        'fvg_overview.html',
        data=data,
        results=results,
        pairs=pairs,
        pair_filter=pair_filter,
        sort_by=sort_by,
        sort_dir=sort_dir,
        variations=FVG_VARIATIONS,
        timestamp=data.get('timestamp', 'N/A'),
        source=data.get('source', 'unknown')
    )


@app.route('/fvg/charts/<pair>')
def fvg_charts(pair):
    """Interactive FVG charts for a specific pair"""
    # Handle URL encoding
    pair = pair.replace('%3D', '=')
    
    variation = request.args.get('variation', None)
    chart_html = create_fvg_chart(pair, variation)
    
    if chart_html is None:
        chart_html = '<div class="error">Failed to load data for this pair</div>'
    
    return render_template(
        'fvg_charts.html',
        pair=pair,
        chart_html=chart_html,
        variations=FVG_VARIATIONS,
        selected_variation=variation,
        pairs=FOREX_PAIRS
    )


@app.route('/fvg/equity')
def fvg_equity():
    """Equity curve comparison page"""
    data = load_fvg_results()
    results = data.get('results', [])
    
    # Get unique pairs for tabs
    pairs = list(set(r.get('symbol', '') for r in results))
    pairs.sort()
    
    pair_filter = request.args.get('pair', '')
    
    equity_chart = create_equity_chart(results, pair_filter if pair_filter else None)
    drawdown_chart = create_drawdown_chart(results, pair_filter if pair_filter else None)
    
    return render_template(
        'fvg_equity.html',
        pairs=pairs,
        pair_filter=pair_filter,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        variations=FVG_VARIATIONS
    )


@app.route('/fvg/api/results')
def api_results():
    """API endpoint for FVG results"""
    data = load_fvg_results()
    return jsonify(data)


@app.route('/fvg/api/chart/<pair>')
def api_chart(pair):
    """API endpoint to get chart data"""
    pair = pair.replace('%3D', '=')
    df = fetch_ohlc_data(pair)
    
    if df is None:
        return jsonify({'error': 'Failed to fetch data'}), 404
    
    fvg_zones = detect_fvg_zones(df)
    
    return jsonify({
        'symbol': pair,
        'ohlc': df.to_dict(orient='records'),
        'fvg_zones': fvg_zones
    })


@app.route('/')
def index():
    """Redirect to FVG overview"""
    return '<script>window.location.href="/fvg"</script>'


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("🎯 FVG Visualization Dashboard")
    print("="*60)
    print(f"📊 Data source: {FVG_RESULTS_FILE}")
    print(f"📊 Fallback:    {BACKTEST_RESULTS_FILE}")
    print()
    print("🌐 Starting server on http://localhost:8889/fvg")
    print()
    print("Pages:")
    print("  /fvg           - FVG Research Overview")
    print("  /fvg/charts/<pair> - Interactive Charts")
    print("  /fvg/equity    - Equity Curve Comparison")
    print("="*60)
    
    app.run(host='0.0.0.0', port=8889, debug=True)
