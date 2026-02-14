"""
Comprehensive Backtest - All Strategies v2
-------------------------------------------
Tests ALL strategies from strategies.py and strategies_forex.py
on forex, stocks, and commodities using the FIXED engine.

No look-ahead bias: uses pending_signal pattern.
"""

import json
from datetime import datetime
from typing import Dict, List, Callable
import pandas as pd
import numpy as np

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import (
    williams_r_signal,
    breakout_signal,
    ma_crossover_signal,
    rsi_divergence_signal,
    fvg_signal,
    volatility_squeeze_signal,
)
from strategies_forex import (
    williams_r_forex,
    breakout_forex,
    ma_crossover_forex,
    session_breakout_forex,
    swing_forex,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# All strategies to test
STRATEGIES = {
    # From strategies.py (generic)
    'williams_r': williams_r_signal,
    'breakout': breakout_signal,
    'ma_crossover': ma_crossover_signal,
    'rsi_divergence': rsi_divergence_signal,
    'fvg': fvg_signal,
    'volatility_squeeze': volatility_squeeze_signal,
    # From strategies_forex.py (forex-optimized)
    'williams_r_fx': williams_r_forex,
    'breakout_fx': breakout_forex,
    'ma_crossover_fx': ma_crossover_forex,
    'session_breakout_fx': session_breakout_forex,
    'swing_fx': swing_forex,
}

# Assets by category
ASSETS = {
    'Forex': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'EURJPY=X'],
    'Stocks': ['SPY', 'QQQ', 'AAPL', 'TSLA'],
    'Commodities': ['GC=F', 'CL=F'],
}

# Commission rates by asset class
COMMISSIONS = {
    'Forex': 0.0003,      # 0.03% (spread cost)
    'Stocks': 0.001,      # 0.1%
    'Commodities': 0.0005, # 0.05%
}


def get_asset_category(symbol: str) -> str:
    """Determine asset category from symbol."""
    for category, symbols in ASSETS.items():
        if symbol in symbols:
            return category
    return 'Stocks'  # Default


def run_backtest(symbol: str, strategy_name: str, strategy_fn: Callable,
                 days: int = 59, interval: str = '1h') -> Dict:
    """Run a single backtest and return metrics."""
    try:
        # Fetch data
        df = fetch_data(symbol, interval=interval, days=days)
        
        if df.empty or len(df) < 50:
            return {
                'symbol': symbol,
                'strategy': strategy_name,
                'status': 'error',
                'error': 'Insufficient data',
            }
        
        # Configure based on asset class
        category = get_asset_category(symbol)
        config = BacktestConfig(
            slippage_pct=0.0005,
            commission_pct=COMMISSIONS.get(category, 0.001),
            position_size=1.0,
            compound=True,
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        trades_df = engine.run(df, strategy_fn)
        metrics = calculate_metrics(trades_df, df, config)
        
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'category': category,
            'status': 'success',
            'data_points': len(df),
            **metrics
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'status': 'error',
            'error': str(e),
        }


def run_all_backtests() -> List[Dict]:
    """Run all strategy/asset combinations."""
    results = []
    
    # Flatten assets
    all_symbols = []
    for category, symbols in ASSETS.items():
        all_symbols.extend(symbols)
    
    total = len(all_symbols) * len(STRATEGIES)
    current = 0
    
    print(f"Running {total} backtests...")
    print("=" * 60)
    
    for symbol in all_symbols:
        category = get_asset_category(symbol)
        print(f"\n{symbol} ({category}):")
        
        for strategy_name, strategy_fn in STRATEGIES.items():
            current += 1
            print(f"  [{current}/{total}] {strategy_name}...", end=" ", flush=True)
            
            result = run_backtest(symbol, strategy_name, strategy_fn)
            results.append(result)
            
            if result['status'] == 'success':
                ret = result['strategy_return']
                wr = result['win_rate']
                trades = result['total_trades']
                status = "✓" if ret > 0 else "✗"
                print(f"{status} Return: {ret:+.1f}%, WR: {wr:.0f}%, Trades: {trades}")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')}")
    
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze results and find best combinations."""
    
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        return {'error': 'No successful backtests'}
    
    # Sort by return
    by_return = sorted(successful, key=lambda x: x['strategy_return'], reverse=True)
    
    # Sort by profit factor (filter infinity)
    by_pf = sorted(
        [r for r in successful if r['profit_factor'] < 900],
        key=lambda x: x['profit_factor'],
        reverse=True
    )
    
    # Sort by Sharpe
    by_sharpe = sorted(
        [r for r in successful if r['sharpe'] != 0],
        key=lambda x: x['sharpe'],
        reverse=True
    )
    
    # Profitable combinations
    profitable = [r for r in successful if r['strategy_return'] > 0]
    
    # By category
    by_category = {}
    for category in ASSETS.keys():
        cat_results = [r for r in successful if r.get('category') == category]
        profitable_cat = [r for r in cat_results if r['strategy_return'] > 0]
        by_category[category] = {
            'total': len(cat_results),
            'profitable': len(profitable_cat),
            'best': max(cat_results, key=lambda x: x['strategy_return']) if cat_results else None,
        }
    
    # By strategy
    by_strategy = {}
    for strategy_name in STRATEGIES.keys():
        strat_results = [r for r in successful if r['strategy'] == strategy_name]
        profitable_strat = [r for r in strat_results if r['strategy_return'] > 0]
        avg_return = np.mean([r['strategy_return'] for r in strat_results]) if strat_results else 0
        by_strategy[strategy_name] = {
            'total': len(strat_results),
            'profitable': len(profitable_strat),
            'avg_return': round(avg_return, 2),
            'best_asset': max(strat_results, key=lambda x: x['strategy_return'])['symbol'] if strat_results else None,
        }
    
    return {
        'summary': {
            'total_backtests': len(successful),
            'profitable_count': len(profitable),
            'profitable_pct': round(len(profitable) / len(successful) * 100, 1) if successful else 0,
        },
        'top_10_by_return': [
            {
                'symbol': r['symbol'],
                'strategy': r['strategy'],
                'return': r['strategy_return'],
                'win_rate': r['win_rate'],
                'trades': r['total_trades'],
                'max_dd': r['max_drawdown'],
            }
            for r in by_return[:10]
        ],
        'top_5_by_profit_factor': [
            {
                'symbol': r['symbol'],
                'strategy': r['strategy'],
                'pf': r['profit_factor'],
                'return': r['strategy_return'],
                'trades': r['total_trades'],
            }
            for r in by_pf[:5]
        ],
        'top_5_by_sharpe': [
            {
                'symbol': r['symbol'],
                'strategy': r['strategy'],
                'sharpe': r['sharpe'],
                'return': r['strategy_return'],
                'trades': r['total_trades'],
            }
            for r in by_sharpe[:5]
        ],
        'by_category': by_category,
        'by_strategy': by_strategy,
    }


def generate_markdown_report(results: List[Dict], analysis: Dict) -> str:
    """Generate markdown summary report."""
    
    lines = [
        "# Curupira Backtests V2 - Results Summary",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Overview",
        "",
        f"- **Total Backtests:** {analysis['summary']['total_backtests']}",
        f"- **Profitable Combinations:** {analysis['summary']['profitable_count']} ({analysis['summary']['profitable_pct']}%)",
        "",
        "### Test Parameters",
        "- Period: 59 days hourly data",
        "- Slippage: 0.05%",
        "- Commission: 0.03% (forex), 0.05% (commodities), 0.1% (stocks)",
        "- Compounding: Yes",
        "- Engine: FIXED (no look-ahead bias)",
        "",
        "---",
        "",
        "## 🏆 Top 10 by Return",
        "",
        "| Rank | Symbol | Strategy | Return % | Win Rate | Trades | Max DD |",
        "|------|--------|----------|----------|----------|--------|--------|",
    ]
    
    for i, r in enumerate(analysis['top_10_by_return'], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['strategy']} | "
            f"{r['return']:+.1f}% | {r['win_rate']:.0f}% | {r['trades']} | {r['max_dd']:.1f}% |"
        )
    
    lines.extend([
        "",
        "## 📊 Top 5 by Profit Factor",
        "",
        "| Symbol | Strategy | PF | Return % | Trades |",
        "|--------|----------|-----|----------|--------|",
    ])
    
    for r in analysis['top_5_by_profit_factor']:
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['pf']:.2f} | {r['return']:+.1f}% | {r['trades']} |"
        )
    
    lines.extend([
        "",
        "## 📈 Top 5 by Sharpe Ratio",
        "",
        "| Symbol | Strategy | Sharpe | Return % | Trades |",
        "|--------|----------|--------|----------|--------|",
    ])
    
    for r in analysis['top_5_by_sharpe']:
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['sharpe']:.2f} | {r['return']:+.1f}% | {r['trades']} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Results by Asset Category",
        "",
    ])
    
    for category, data in analysis['by_category'].items():
        lines.append(f"### {category}")
        lines.append(f"- Tested: {data['total']} combinations")
        lines.append(f"- Profitable: {data['profitable']}")
        if data['best']:
            best = data['best']
            lines.append(f"- **Best:** {best['symbol']} + {best['strategy']} = {best['strategy_return']:+.1f}%")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Results by Strategy",
        "",
        "| Strategy | Assets Tested | Profitable | Avg Return | Best Asset |",
        "|----------|--------------|------------|------------|------------|",
    ])
    
    for strategy, data in analysis['by_strategy'].items():
        lines.append(
            f"| {strategy} | {data['total']} | {data['profitable']} | "
            f"{data['avg_return']:+.1f}% | {data['best_asset'] or 'N/A'} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Key Insights",
        "",
        "### What Works",
        "",
    ])
    
    # Find strategies with >50% profitable rate
    good_strategies = [
        (name, data) for name, data in analysis['by_strategy'].items()
        if data['total'] > 0 and data['profitable'] / data['total'] >= 0.5
    ]
    
    if good_strategies:
        for name, data in sorted(good_strategies, key=lambda x: x[1]['avg_return'], reverse=True):
            pct = data['profitable'] / data['total'] * 100
            lines.append(f"- **{name}**: {pct:.0f}% profitable across assets, avg return {data['avg_return']:+.1f}%")
    else:
        lines.append("- No strategies showed >50% profitability across all tested assets")
    
    lines.extend([
        "",
        "### What Doesn't Work",
        "",
    ])
    
    # Find strategies with 0 profitable
    bad_strategies = [
        name for name, data in analysis['by_strategy'].items()
        if data['total'] > 0 and data['profitable'] == 0
    ]
    
    if bad_strategies:
        lines.append(f"- Strategies with 0 profitable combinations: {', '.join(bad_strategies)}")
    else:
        lines.append("- All strategies had at least some profitable combinations")
    
    lines.extend([
        "",
        "---",
        "",
        "## Full Results",
        "",
        "See `backtest_results_v2.json` for complete data.",
        "",
    ])
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("CURUPIRA BACKTESTS V2 - COMPREHENSIVE TEST")
    print("Engine: FIXED (no look-ahead bias)")
    print("=" * 60)
    
    # Run all backtests
    results = run_all_backtests()
    
    # Analyze
    print("\n\nAnalyzing results...")
    analysis = analyze_results(results)
    
    # Save JSON results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'days': 59,
            'interval': '1h',
            'slippage': 0.0005,
        },
        'strategies': list(STRATEGIES.keys()),
        'assets': ASSETS,
        'results': results,
        'analysis': analysis,
    }
    
    with open('backtest_results_v2.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to backtest_results_v2.json")
    
    # Generate markdown report
    report = generate_markdown_report(results, analysis)
    with open('RESULTS_V2.md', 'w') as f:
        f.write(report)
    
    print(f"Summary saved to RESULTS_V2.md")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal: {analysis['summary']['total_backtests']} backtests")
    print(f"Profitable: {analysis['summary']['profitable_count']} ({analysis['summary']['profitable_pct']}%)")
    
    print("\nTop 5 by Return:")
    for i, r in enumerate(analysis['top_10_by_return'][:5], 1):
        print(f"  {i}. {r['symbol']} + {r['strategy']}: {r['return']:+.1f}%")
    
    print("\nBy Strategy (avg return):")
    sorted_strategies = sorted(
        analysis['by_strategy'].items(),
        key=lambda x: x[1]['avg_return'],
        reverse=True
    )
    for name, data in sorted_strategies[:5]:
        print(f"  {name}: {data['avg_return']:+.1f}% ({data['profitable']}/{data['total']} profitable)")


if __name__ == "__main__":
    main()
