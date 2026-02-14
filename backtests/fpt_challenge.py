#!/usr/bin/env python3
"""
First-Passage Time Prop Firm Challenge Optimizer
================================================

Based on: Perelló et al. "Scaling properties and universality of 
first-passage time probabilities in financial markets"

Key insight: You don't need alpha. You need SURVIVAL.

The challenge is a race between two barriers:
- Profit target (+10%)
- Drawdown limit (-5% daily, -10% total)

The math tells us:
- P(hit target) depends on volatility and time
- P(survive) depends on position sizing
- Optimal strategy = maximize P(profit) × P(survive)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ChallengeParams:
    """Prop firm challenge parameters."""
    initial_balance: float = 100_000
    profit_target_pct: float = 0.10      # 10%
    max_daily_dd_pct: float = 0.05       # 5%
    max_total_dd_pct: float = 0.10       # 10%
    time_limit_days: int = 30
    challenge_cost: float = 540          # EUR


@dataclass
class MarketParams:
    """Market volatility parameters."""
    daily_vol_pct: float                 # Daily volatility (e.g., 0.01 for 1%)
    alpha: float = 3.32                  # Universal tail exponent
    a: float = 15e-4                     # Scale parameter (s^-1)
    name: str = "Unknown"


# Pre-calibrated market parameters
MARKETS = {
    'EURUSD': MarketParams(daily_vol_pct=0.005, alpha=3.49, a=24.9e-4, name='EUR/USD'),
    'GBPUSD': MarketParams(daily_vol_pct=0.006, alpha=3.32, a=20e-4, name='GBP/USD'),
    'USDJPY': MarketParams(daily_vol_pct=0.005, alpha=3.32, a=18e-4, name='USD/JPY'),
    'XAUUSD': MarketParams(daily_vol_pct=0.012, alpha=3.32, a=15e-4, name='Gold'),
    'BTCUSD': MarketParams(daily_vol_pct=0.035, alpha=3.32, a=10e-4, name='Bitcoin'),
    'ETHUSD': MarketParams(daily_vol_pct=0.045, alpha=3.32, a=10e-4, name='Ethereum'),
    'LINKUSD': MarketParams(daily_vol_pct=0.055, alpha=3.32, a=10e-4, name='Chainlink'),
}


def fpt_probability(x_sigma: float, t_seconds: float, alpha: float, a: float) -> float:
    """
    First-passage time probability using Student distribution.
    
    P(first cross level x before time t)
    
    Args:
        x_sigma: Target in volatility units (target% / daily_vol%)
        t_seconds: Time horizon in seconds
        alpha: Tail exponent (universal ≈ 3.32)
        a: Scale parameter (market dependent)
    
    Returns:
        Probability [0, 1]
    """
    if t_seconds <= 0 or a <= 0:
        return 0.0
    return float((1 + x_sigma / np.sqrt(a * t_seconds)) ** (-alpha))


def survival_probability(dd_sigma: float, t_seconds: float, alpha: float, a: float) -> float:
    """
    Probability of NOT hitting the drawdown limit before time t.
    
    S(x, t) = 1 - W(x, t)
    """
    return 1.0 - fpt_probability(dd_sigma, t_seconds, alpha, a)


def target_to_sigma(target_pct: float, daily_vol_pct: float) -> float:
    """Convert percentage target to volatility units."""
    return target_pct / daily_vol_pct


def challenge_probability(
    params: ChallengeParams,
    market: MarketParams,
    position_size: float = 1.0,
    leverage: float = 1.0
) -> dict:
    """
    Calculate probability of passing the challenge.
    
    Args:
        params: Challenge parameters
        market: Market volatility parameters
        position_size: Fraction of capital per trade (0-1)
        leverage: Trading leverage multiplier
    
    Returns:
        Dict with probabilities and analysis
    """
    # Time in seconds
    t_seconds = params.time_limit_days * 24 * 60 * 60
    
    # Effective volatility (scaled by position size and leverage)
    effective_vol = market.daily_vol_pct * position_size * leverage
    
    # Targets in sigma units (using effective vol)
    profit_sigma = target_to_sigma(params.profit_target_pct, effective_vol)
    daily_dd_sigma = target_to_sigma(params.max_daily_dd_pct, effective_vol)
    total_dd_sigma = target_to_sigma(params.max_total_dd_pct, effective_vol)
    
    # Probabilities
    p_hit_profit = fpt_probability(profit_sigma, t_seconds, market.alpha, market.a)
    p_survive_daily = survival_probability(daily_dd_sigma, t_seconds, market.alpha, market.a)
    p_survive_total = survival_probability(total_dd_sigma, t_seconds, market.alpha, market.a)
    
    # Combined survival (conservative: use the tighter constraint)
    p_survive = min(p_survive_daily, p_survive_total)
    
    # Simplified pass probability (not exact but directional)
    # In reality these are correlated, but this gives intuition
    p_pass = p_hit_profit * p_survive
    
    return {
        'market': market.name,
        'position_size': position_size,
        'leverage': leverage,
        'effective_daily_vol': effective_vol,
        'profit_sigma': profit_sigma,
        'dd_sigma': daily_dd_sigma,
        'p_hit_profit': p_hit_profit,
        'p_survive_daily': p_survive_daily,
        'p_survive_total': p_survive_total,
        'p_survive': p_survive,
        'p_pass_approx': p_pass,
    }


def expected_value(
    params: ChallengeParams,
    market: MarketParams,
    position_size: float = 1.0,
    leverage: float = 1.0,
    funded_profit_split: float = 0.80,
    expected_monthly_return: float = 0.05
) -> dict:
    """
    Calculate expected value of attempting the challenge.
    
    Args:
        params: Challenge parameters
        market: Market parameters
        position_size: Position sizing
        leverage: Leverage
        funded_profit_split: Your share of profits when funded (80%)
        expected_monthly_return: Expected monthly return when funded (5%)
    
    Returns:
        Dict with EV analysis
    """
    prob = challenge_probability(params, market, position_size, leverage)
    
    # If you pass, expected profit from funded account (simplified)
    # Assume you trade for average 6 months before blowing up or quitting
    funded_months = 6
    funded_profit = params.initial_balance * expected_monthly_return * funded_months * funded_profit_split
    
    # Expected value
    ev_pass = prob['p_pass_approx'] * funded_profit
    ev_fail = (1 - prob['p_pass_approx']) * 0  # Lose challenge fee
    ev_total = ev_pass - params.challenge_cost
    
    # Break-even pass rate needed
    break_even_pass_rate = params.challenge_cost / funded_profit if funded_profit > 0 else 1.0
    
    return {
        **prob,
        'challenge_cost': params.challenge_cost,
        'funded_profit_potential': funded_profit,
        'ev_total': ev_total,
        'break_even_pass_rate': break_even_pass_rate,
        'ev_per_attempt': ev_total,
        'roi_if_positive': (ev_total / params.challenge_cost * 100) if ev_total > 0 else 0,
    }


def optimize_position_size(
    params: ChallengeParams,
    market: MarketParams,
    leverage: float = 1.0,
    min_size: float = 0.05,
    max_size: float = 1.0,
    steps: int = 20
) -> Tuple[float, dict]:
    """
    Find optimal position size that maximizes pass probability.
    
    Returns:
        Tuple of (optimal_size, results_dict)
    """
    best_size = min_size
    best_result = None
    best_pass_prob = 0
    
    results = []
    for size in np.linspace(min_size, max_size, steps):
        result = challenge_probability(params, market, size, leverage)
        results.append(result)
        
        if result['p_pass_approx'] > best_pass_prob:
            best_pass_prob = result['p_pass_approx']
            best_size = size
            best_result = result
    
    return best_size, best_result, results


def monte_carlo_challenge(
    params: ChallengeParams,
    market: MarketParams,
    position_size: float = 1.0,
    leverage: float = 1.0,
    n_simulations: int = 10000,
    trades_per_day: int = 2,
    win_rate: float = 0.50,  # No alpha = coin flip
    risk_reward: float = 2.0,
) -> dict:
    """
    Monte Carlo simulation of the challenge.
    
    This uses actual trade simulation rather than the closed-form FPT.
    Useful for validating the analytical results.
    
    Args:
        params: Challenge parameters
        market: Market parameters
        position_size: Risk per trade as fraction of account
        leverage: Leverage multiplier
        n_simulations: Number of challenge simulations
        trades_per_day: Average trades per day
        win_rate: Win rate (0.5 = no edge, just variance)
        risk_reward: Risk:Reward ratio (2.0 = 2:1)
    """
    np.random.seed(42)
    
    n_trades = params.time_limit_days * trades_per_day
    risk_per_trade = position_size * leverage * market.daily_vol_pct / trades_per_day
    
    # Results tracking
    passed = 0
    failed_daily_dd = 0
    failed_total_dd = 0
    failed_timeout = 0
    final_balances = []
    
    for _ in range(n_simulations):
        balance = params.initial_balance
        peak_balance = balance
        daily_start = balance
        
        day = 0
        hit_profit = False
        hit_dd = False
        
        for t in range(n_trades):
            # New day reset
            if t % trades_per_day == 0:
                daily_start = balance
                day += 1
            
            # Generate trade outcome
            if np.random.random() < win_rate:
                # Win: gain R×risk
                pnl = balance * risk_per_trade * risk_reward
            else:
                # Loss: lose risk amount
                pnl = -balance * risk_per_trade
            
            balance += pnl
            peak_balance = max(peak_balance, balance)
            
            # Check daily drawdown
            daily_dd = (daily_start - balance) / params.initial_balance
            if daily_dd >= params.max_daily_dd_pct:
                failed_daily_dd += 1
                hit_dd = True
                break
            
            # Check total drawdown
            total_dd = (peak_balance - balance) / params.initial_balance
            if total_dd >= params.max_total_dd_pct:
                failed_total_dd += 1
                hit_dd = True
                break
            
            # Check profit target
            profit = (balance - params.initial_balance) / params.initial_balance
            if profit >= params.profit_target_pct:
                passed += 1
                hit_profit = True
                break
        
        if not hit_profit and not hit_dd:
            # Timeout - didn't hit either barrier
            if balance >= params.initial_balance * (1 + params.profit_target_pct):
                passed += 1
            else:
                failed_timeout += 1
        
        final_balances.append(balance)
    
    pass_rate = passed / n_simulations
    
    return {
        'n_simulations': n_simulations,
        'position_size': position_size,
        'risk_per_trade': risk_per_trade,
        'win_rate': win_rate,
        'risk_reward': risk_reward,
        'passed': passed,
        'pass_rate': pass_rate,
        'failed_daily_dd': failed_daily_dd,
        'failed_total_dd': failed_total_dd,
        'failed_timeout': failed_timeout,
        'avg_final_balance': np.mean(final_balances),
        'median_final_balance': np.median(final_balances),
    }


def print_analysis(market_name: str = 'EURUSD'):
    """Print comprehensive analysis for a market."""
    params = ChallengeParams()
    market = MARKETS.get(market_name, MARKETS['EURUSD'])
    
    print("=" * 60)
    print(f"PROP FIRM CHALLENGE ANALYSIS: {market.name}")
    print("=" * 60)
    print(f"\nChallenge: ${params.initial_balance:,.0f}")
    print(f"Profit Target: {params.profit_target_pct:.0%}")
    print(f"Max Daily DD: {params.max_daily_dd_pct:.0%}")
    print(f"Time Limit: {params.time_limit_days} days")
    print(f"Cost: €{params.challenge_cost}")
    
    print(f"\nMarket: {market.name}")
    print(f"Daily Volatility: {market.daily_vol_pct:.1%}")
    print(f"Tail Exponent (α): {market.alpha}")
    
    # Analyze different position sizes
    print("\n" + "-" * 60)
    print("POSITION SIZE ANALYSIS (FPT Model)")
    print("-" * 60)
    print(f"{'Size':>8} {'P(Profit)':>10} {'P(Survive)':>11} {'P(Pass)':>10} {'Sigma':>8}")
    print("-" * 60)
    
    for size in [0.10, 0.25, 0.50, 0.75, 1.00]:
        result = challenge_probability(params, market, position_size=size)
        print(f"{size:>7.0%} {result['p_hit_profit']:>10.1%} {result['p_survive']:>11.1%} "
              f"{result['p_pass_approx']:>10.1%} {result['profit_sigma']:>8.1f}σ")
    
    # Find optimal
    opt_size, opt_result, _ = optimize_position_size(params, market)
    print("-" * 60)
    print(f"OPTIMAL: {opt_size:.0%} position size -> {opt_result['p_pass_approx']:.1%} pass rate")
    
    # Monte Carlo validation
    print("\n" + "-" * 60)
    print("MONTE CARLO VALIDATION (10k sims, no alpha)")
    print("-" * 60)
    
    for size in [0.10, 0.25, opt_size]:
        mc = monte_carlo_challenge(params, market, position_size=size, 
                                   win_rate=0.50, risk_reward=2.0)
        print(f"Size {size:.0%}: Pass={mc['pass_rate']:.1%}, "
              f"DD_fail={mc['failed_daily_dd']/mc['n_simulations']:.1%}, "
              f"Timeout={mc['failed_timeout']/mc['n_simulations']:.1%}")
    
    # Expected Value
    print("\n" + "-" * 60)
    print("EXPECTED VALUE (with optimal sizing)")
    print("-" * 60)
    ev = expected_value(params, market, position_size=opt_size)
    print(f"Challenge Cost: €{ev['challenge_cost']}")
    print(f"If Pass, Funded Profit Potential: €{ev['funded_profit_potential']:,.0f}")
    print(f"Pass Probability: {ev['p_pass_approx']:.1%}")
    print(f"Expected Value per Attempt: €{ev['ev_total']:,.0f}")
    print(f"Break-even Pass Rate Needed: {ev['break_even_pass_rate']:.1%}")
    
    if ev['ev_total'] > 0:
        print(f"\n✅ POSITIVE EV: {ev['roi_if_positive']:.0f}% ROI per attempt")
    else:
        print(f"\n❌ NEGATIVE EV: Need better edge or different market")


def compare_markets():
    """Compare all markets."""
    params = ChallengeParams()
    
    print("=" * 70)
    print("MARKET COMPARISON (Optimal Position Size)")
    print("=" * 70)
    print(f"{'Market':<12} {'Vol':>6} {'Opt Size':>9} {'P(Pass)':>9} {'EV':>10}")
    print("-" * 70)
    
    for name, market in MARKETS.items():
        opt_size, opt_result, _ = optimize_position_size(params, market)
        ev = expected_value(params, market, position_size=opt_size)
        print(f"{market.name:<12} {market.daily_vol_pct:>5.1%} {opt_size:>8.0%} "
              f"{opt_result['p_pass_approx']:>9.1%} €{ev['ev_total']:>9,.0f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        market = sys.argv[1].upper()
        if market in MARKETS:
            print_analysis(market)
        elif market == 'ALL':
            compare_markets()
        else:
            print(f"Unknown market: {market}")
            print(f"Available: {', '.join(MARKETS.keys())}, ALL")
    else:
        print_analysis('EURUSD')
        print("\n")
        compare_markets()
