"""
Prop Firm Challenge Optimizer
=============================
Not looking for alpha. Optimizing first-passage time problem.

Goal: Maximize P(hit +10% before hitting -10%)

This is a gambler's ruin / random walk with absorbing barriers problem.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json


def first_passage_probability(
    win_rate: float,
    win_size: float,  # As fraction of account (e.g., 0.02 = 2%)
    loss_size: float,  # As fraction of account (e.g., 0.01 = 1%)
    target: float = 0.10,  # +10%
    max_dd: float = 0.10,  # -10%
    n_simulations: int = 10000
) -> Dict:
    """
    Monte Carlo simulation of first-passage time problem.
    
    Returns probability of hitting target before hitting max DD.
    """
    passed = 0
    blown = 0
    trade_counts = []
    
    for _ in range(n_simulations):
        equity = 0.0  # Start at 0% P&L
        trades = 0
        
        while True:
            trades += 1
            
            # Simulate trade
            if np.random.random() < win_rate:
                equity += win_size
            else:
                equity -= loss_size
            
            # Check barriers
            if equity >= target:
                passed += 1
                trade_counts.append(trades)
                break
            elif equity <= -max_dd:
                blown += 1
                trade_counts.append(trades)
                break
            
            # Safety limit
            if trades > 1000:
                blown += 1
                break
    
    pass_rate = passed / n_simulations
    
    return {
        'pass_rate': round(pass_rate * 100, 2),
        'avg_trades_to_resolution': round(np.mean(trade_counts), 1),
        'median_trades': round(np.median(trade_counts), 0),
    }


def optimize_position_size(
    win_rate: float,
    rr_ratio: float,  # Risk:Reward (e.g., 2.0 means 1:2)
    target: float = 0.10,
    max_dd: float = 0.10,
    daily_dd: float = 0.05,
    n_simulations: int = 5000
) -> Dict:
    """
    Find optimal position size (risk per trade) that maximizes pass probability.
    
    Constraints:
    - Can't risk more than daily DD limit on single trade
    - Can't exceed total DD limit
    """
    results = []
    
    # Test different risk sizes (as % of account)
    for risk_pct in np.arange(0.005, daily_dd + 0.005, 0.005):  # 0.5% to 5%
        risk_pct = round(risk_pct, 3)
        reward_pct = risk_pct * rr_ratio
        
        sim = first_passage_probability(
            win_rate=win_rate,
            win_size=reward_pct,
            loss_size=risk_pct,
            target=target,
            max_dd=max_dd,
            n_simulations=n_simulations
        )
        
        results.append({
            'risk_pct': risk_pct * 100,
            'reward_pct': reward_pct * 100,
            **sim
        })
    
    # Find optimal
    best = max(results, key=lambda x: x['pass_rate'])
    
    return {
        'optimal': best,
        'all_results': results,
    }


def event_concentration_analysis(
    win_rate: float,
    risk_pct: float,
    rr_ratio: float,
    n_events: int,  # Number of high-vol events to trade
    n_simulations: int = 5000
) -> Dict:
    """
    Simulate trading only N high-volatility events.
    
    Hypothesis: Fewer trades with higher conviction = better pass rate
    """
    reward_pct = risk_pct * rr_ratio
    target = 0.10
    max_dd = 0.10
    
    passed = 0
    
    for _ in range(n_simulations):
        equity = 0.0
        
        for _ in range(n_events):
            if np.random.random() < win_rate:
                equity += reward_pct
            else:
                equity -= risk_pct
            
            # Check barriers after each trade
            if equity >= target:
                passed += 1
                break
            elif equity <= -max_dd:
                break
    
    return {
        'n_events': n_events,
        'pass_rate': round(passed / n_simulations * 100, 2),
    }


def ev_per_attempt(
    pass_rate: float,  # As decimal
    challenge_fee: float,
    expected_extraction: float  # Expected profit if funded
) -> float:
    """Calculate expected value per challenge attempt."""
    return (pass_rate * expected_extraction) - challenge_fee


def run_analysis():
    """Full prop firm challenge optimization analysis."""
    
    print("="*70)
    print("PROP FIRM CHALLENGE OPTIMIZER")
    print("First-passage time problem with absorbing barriers")
    print("="*70)
    
    # ==========================================================================
    # 1. BASELINE: Random trading (50% WR)
    # ==========================================================================
    print("\n" + "="*70)
    print("1. BASELINE: Random 50% win rate")
    print("="*70)
    
    for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        opt = optimize_position_size(win_rate=0.50, rr_ratio=rr, n_simulations=10000)
        best = opt['optimal']
        print(f"  RR 1:{rr:.1f} → Optimal risk={best['risk_pct']:.1f}% | Pass rate={best['pass_rate']:.1f}% | Avg trades={best['avg_trades_to_resolution']:.0f}")
    
    # ==========================================================================
    # 2. EDGE SCENARIOS: Slight edge compounds
    # ==========================================================================
    print("\n" + "="*70)
    print("2. SLIGHT EDGE IMPACT (1:2 RR)")
    print("="*70)
    
    for wr in [0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        opt = optimize_position_size(win_rate=wr, rr_ratio=2.0, n_simulations=10000)
        best = opt['optimal']
        print(f"  WR={wr*100:.0f}% → Optimal risk={best['risk_pct']:.1f}% | Pass rate={best['pass_rate']:.1f}%")
    
    # ==========================================================================
    # 3. EVENT CONCENTRATION: Fewer trades, higher stakes
    # ==========================================================================
    print("\n" + "="*70)
    print("3. EVENT CONCENTRATION (50% WR, 1:2 RR)")
    print("="*70)
    
    # If we go all-in on 5% risk (daily DD limit) with 10% reward (1:2)
    # We need 1 win to hit target, 2 losses to blow out
    for n_events in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        result = event_concentration_analysis(
            win_rate=0.50,
            risk_pct=0.05,  # 5% risk (max daily DD)
            rr_ratio=2.0,
            n_events=n_events,
            n_simulations=10000
        )
        print(f"  {n_events:2} events → Pass rate={result['pass_rate']:.1f}%")
    
    # ==========================================================================
    # 4. OPTIMAL EVENT COUNT BY WIN RATE
    # ==========================================================================
    print("\n" + "="*70)
    print("4. OPTIMAL EVENT COUNT BY WIN RATE (5% risk, 1:2 RR)")
    print("="*70)
    
    for wr in [0.45, 0.50, 0.55, 0.60]:
        best_n = 0
        best_pass = 0
        for n in range(2, 21):
            result = event_concentration_analysis(wr, 0.05, 2.0, n, 10000)
            if result['pass_rate'] > best_pass:
                best_pass = result['pass_rate']
                best_n = n
        print(f"  WR={wr*100:.0f}% → Optimal events={best_n} | Pass rate={best_pass:.1f}%")
    
    # ==========================================================================
    # 5. EV CALCULATION (FTMO €100k)
    # ==========================================================================
    print("\n" + "="*70)
    print("5. EXPECTED VALUE PER ATTEMPT (FTMO €100k)")
    print("="*70)
    
    fee = 540  # €540
    extractions = [5000, 10000, 15000]  # Expected extraction if funded
    
    for extraction in extractions:
        print(f"\n  Assuming €{extraction} extraction if funded:")
        for pass_rate in [0.10, 0.15, 0.20, 0.25, 0.30]:
            ev = ev_per_attempt(pass_rate, fee, extraction)
            print(f"    {pass_rate*100:.0f}% pass rate → EV = €{ev:+.0f}")
    
    # ==========================================================================
    # 6. SERIAL ATTEMPTS: How many to profitability?
    # ==========================================================================
    print("\n" + "="*70)
    print("6. SERIAL ATTEMPTS TO PROFITABILITY")
    print("="*70)
    
    pass_rate = 0.20  # 20%
    fee = 540
    extraction = 10000
    
    print(f"  Assumptions: {pass_rate*100:.0f}% pass rate, €{fee} fee, €{extraction} extraction")
    
    # Simulate N attempts until profit
    n_sims = 10000
    attempts_to_profit = []
    total_profit = []
    
    for _ in range(n_sims):
        attempts = 0
        profit = 0
        
        while profit <= 0 and attempts < 100:
            attempts += 1
            profit -= fee
            
            if np.random.random() < pass_rate:
                profit += extraction
        
        attempts_to_profit.append(attempts)
        total_profit.append(profit)
    
    print(f"  Median attempts to net profit: {np.median(attempts_to_profit):.0f}")
    print(f"  Mean attempts: {np.mean(attempts_to_profit):.1f}")
    print(f"  P(profit after 10 attempts): {sum(1 for p in total_profit if p > 0) / n_sims * 100:.1f}%")


if __name__ == '__main__':
    run_analysis()
