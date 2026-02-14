"""
Pure Variance Play for Prop Firm Challenges
============================================
Forget alpha. Exploit the challenge structure.

INSIGHT: FTMO challenge is a first-passage time problem.
- You need to hit +10% before -10%
- With symmetric barriers and no edge, P(pass) = 50%
- But challenge fee is €540 and payout is €10,000+
- Even 20% pass rate = massive +EV

STRATEGY: Maximize variance while respecting daily DD limit.
- Trade high-vol events with controlled risk
- Let the math work in your favor

NO ALPHA NEEDED. Just variance and survival.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from datetime import datetime


@dataclass
class ChallengeConfig:
    """FTMO Challenge Rules"""
    fee: float = 540  # EUR for 100k challenge
    account_size: float = 100_000
    profit_target_pct: float = 10.0
    daily_dd_pct: float = 5.0
    max_dd_pct: float = 10.0
    min_trading_days: int = 4
    challenge_days: int = 30
    
    # Payout structure
    first_payout: float = 10_000  # Conservative estimate after verification
    monthly_payout: float = 8_000  # 80% of ~10k monthly profit


def simulate_challenge(
    risk_per_trade: float,
    rr_ratio: float,
    win_rate: float,
    trades_per_day: float,
    config: ChallengeConfig = None
) -> Dict:
    """
    Simulate a single challenge attempt.
    
    Returns dict with outcome and stats.
    """
    if config is None:
        config = ChallengeConfig()
    
    balance = config.account_size
    high_water = balance
    daily_start = balance
    current_day = 0
    trading_days = set()
    trades_taken = 0
    max_dd_seen = 0
    
    for day in range(config.challenge_days):
        daily_start = balance
        
        # Random number of trades today (Poisson-ish)
        n_trades_today = np.random.poisson(trades_per_day)
        
        if n_trades_today > 0:
            trading_days.add(day)
        
        for _ in range(n_trades_today):
            trades_taken += 1
            
            # Trade outcome
            if np.random.random() < win_rate:
                pnl_pct = risk_per_trade * rr_ratio
            else:
                pnl_pct = -risk_per_trade
            
            balance = balance * (1 + pnl_pct / 100)
            
            # Update high water mark
            if balance > high_water:
                high_water = balance
            
            # Check daily DD (from daily start, measured against INITIAL balance)
            daily_dd = (daily_start - balance) / config.account_size * 100
            if daily_dd >= config.daily_dd_pct:
                return {
                    "passed": False,
                    "fail_reason": "daily_dd",
                    "final_balance": balance,
                    "final_pnl_pct": (balance - config.account_size) / config.account_size * 100,
                    "max_dd_pct": max_dd_seen,
                    "trades": trades_taken,
                    "trading_days": len(trading_days)
                }
            
            # Check max DD (from high water, measured against INITIAL balance)
            max_dd = (high_water - balance) / config.account_size * 100
            max_dd_seen = max(max_dd_seen, max_dd)
            
            if max_dd >= config.max_dd_pct:
                return {
                    "passed": False,
                    "fail_reason": "max_dd",
                    "final_balance": balance,
                    "final_pnl_pct": (balance - config.account_size) / config.account_size * 100,
                    "max_dd_pct": max_dd_seen,
                    "trades": trades_taken,
                    "trading_days": len(trading_days)
                }
            
            # Check if passed
            profit_pct = (balance - config.account_size) / config.account_size * 100
            if profit_pct >= config.profit_target_pct and len(trading_days) >= config.min_trading_days:
                return {
                    "passed": True,
                    "fail_reason": None,
                    "final_balance": balance,
                    "final_pnl_pct": profit_pct,
                    "max_dd_pct": max_dd_seen,
                    "trades": trades_taken,
                    "trading_days": len(trading_days)
                }
    
    # Time expired
    profit_pct = (balance - config.account_size) / config.account_size * 100
    return {
        "passed": False,
        "fail_reason": "time_expired",
        "final_balance": balance,
        "final_pnl_pct": profit_pct,
        "max_dd_pct": max_dd_seen,
        "trades": trades_taken,
        "trading_days": len(trading_days)
    }


def monte_carlo_challenge(
    n_sims: int,
    risk_per_trade: float,
    rr_ratio: float,
    win_rate: float,
    trades_per_day: float,
    config: ChallengeConfig = None
) -> Dict:
    """Run Monte Carlo simulation of challenge attempts."""
    
    if config is None:
        config = ChallengeConfig()
    
    results = {
        "passed": 0,
        "failed_daily_dd": 0,
        "failed_max_dd": 0,
        "failed_time": 0,
        "final_balances": [],
        "max_dds": [],
        "trades_to_pass": []
    }
    
    for _ in range(n_sims):
        outcome = simulate_challenge(
            risk_per_trade=risk_per_trade,
            rr_ratio=rr_ratio,
            win_rate=win_rate,
            trades_per_day=trades_per_day,
            config=config
        )
        
        results["final_balances"].append(outcome["final_balance"])
        results["max_dds"].append(outcome["max_dd_pct"])
        
        if outcome["passed"]:
            results["passed"] += 1
            results["trades_to_pass"].append(outcome["trades"])
        elif outcome["fail_reason"] == "daily_dd":
            results["failed_daily_dd"] += 1
        elif outcome["fail_reason"] == "max_dd":
            results["failed_max_dd"] += 1
        else:
            results["failed_time"] += 1
    
    # Calculate stats
    pass_rate = results["passed"] / n_sims
    
    # EV calculation
    ev_per_attempt = (pass_rate * config.first_payout) - config.fee
    
    # Attempts needed for net profit
    if pass_rate > 0:
        expected_attempts = 1 / pass_rate
        breakeven_attempts = config.fee / (pass_rate * config.first_payout) if pass_rate > 0 else float('inf')
    else:
        expected_attempts = float('inf')
        breakeven_attempts = float('inf')
    
    return {
        "n_sims": n_sims,
        "pass_rate": pass_rate * 100,
        "failed_daily_dd_pct": results["failed_daily_dd"] / n_sims * 100,
        "failed_max_dd_pct": results["failed_max_dd"] / n_sims * 100,
        "failed_time_pct": results["failed_time"] / n_sims * 100,
        "avg_final_balance": np.mean(results["final_balances"]),
        "avg_max_dd": np.mean(results["max_dds"]),
        "ev_per_attempt": ev_per_attempt,
        "expected_attempts_to_pass": expected_attempts,
        "breakeven_attempts": breakeven_attempts,
        "avg_trades_to_pass": np.mean(results["trades_to_pass"]) if results["trades_to_pass"] else 0
    }


def find_optimal_params():
    """Find optimal risk/RR parameters for different win rates."""
    
    print("="*80)
    print("PURE VARIANCE PLAY - OPTIMAL PARAMETER SEARCH")
    print("="*80)
    print("\nAssumption: NO EDGE (50% win rate baseline)")
    print("Goal: Maximize pass rate while respecting daily DD\n")
    
    # Parameter grid
    risks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    rr_ratios = [1.5, 2.0, 2.5, 3.0]
    win_rates = [0.45, 0.50, 0.55]  # Slightly above/below neutral
    trades_per_day_opts = [0.5, 1.0, 2.0]  # Conservative to aggressive
    
    all_results = []
    
    for wr in win_rates:
        print(f"\n{'='*60}")
        print(f"WIN RATE: {wr*100:.0f}%")
        print(f"{'='*60}")
        
        best_for_wr = None
        best_ev = -float('inf')
        
        for risk in risks:
            for rr in rr_ratios:
                for tpd in trades_per_day_opts:
                    result = monte_carlo_challenge(
                        n_sims=2000,
                        risk_per_trade=risk,
                        rr_ratio=rr,
                        win_rate=wr,
                        trades_per_day=tpd
                    )
                    
                    result["params"] = {
                        "risk": risk,
                        "rr": rr,
                        "win_rate": wr,
                        "trades_per_day": tpd
                    }
                    
                    all_results.append(result)
                    
                    if result["ev_per_attempt"] > best_ev:
                        best_ev = result["ev_per_attempt"]
                        best_for_wr = result
        
        if best_for_wr:
            p = best_for_wr["params"]
            print(f"\nBEST CONFIG:")
            print(f"  Risk: {p['risk']}% | RR: {p['rr']}:1 | Trades/day: {p['trades_per_day']}")
            print(f"  Pass Rate: {best_for_wr['pass_rate']:.1f}%")
            print(f"  EV/attempt: €{best_for_wr['ev_per_attempt']:,.0f}")
            print(f"  Avg trades to pass: {best_for_wr['avg_trades_to_pass']:.0f}")
            print(f"  Fail reasons: Daily DD {best_for_wr['failed_daily_dd_pct']:.0f}% | "
                  f"Max DD {best_for_wr['failed_max_dd_pct']:.0f}% | "
                  f"Time {best_for_wr['failed_time_pct']:.0f}%")
    
    # Sort all results by EV
    all_results.sort(key=lambda x: x["ev_per_attempt"], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by EV)")
    print("="*80)
    print(f"{'WR%':<6} {'Risk':<6} {'RR':<5} {'TPD':<5} {'Pass%':<8} {'EV':<10} {'DailyDD%':<10}")
    print("-"*80)
    
    for r in all_results[:10]:
        p = r["params"]
        print(f"{p['win_rate']*100:<6.0f} {p['risk']:<6.1f} {p['rr']:<5.1f} {p['trades_per_day']:<5.1f} "
              f"{r['pass_rate']:<8.1f} €{r['ev_per_attempt']:<9,.0f} {r['failed_daily_dd_pct']:<10.1f}")
    
    # Save results
    with open("variance_play_results.json", "w") as f:
        json.dump(all_results[:50], f, indent=2, default=str)
    
    return all_results


def analyze_crypto_vs_forex():
    """Compare crypto vs forex for variance play."""
    
    print("\n" + "="*80)
    print("CRYPTO VS FOREX FOR VARIANCE PLAY")
    print("="*80)
    
    # Crypto: Higher volatility = more trades hit targets/stops faster
    # Forex: Lower volatility = more timeouts, slower progress
    
    # Simulate with different "volatility profiles"
    # Higher vol = more trades per day equivalent (faster resolution)
    
    scenarios = [
        ("Forex (low vol)", 0.50, 1.0),   # 1 trade/day resolves
        ("Forex (med vol)", 0.50, 2.0),   # 2 trades/day resolve
        ("Crypto (high vol)", 0.50, 4.0), # 4 trades/day resolve
        ("Crypto (meme)", 0.50, 8.0),     # 8 trades/day resolve
    ]
    
    print(f"\n{'Scenario':<20} {'Pass%':<8} {'EV':<10} {'Avg Days':<10} {'DailyDD%':<10}")
    print("-"*60)
    
    for name, wr, tpd in scenarios:
        result = monte_carlo_challenge(
            n_sims=3000,
            risk_per_trade=2.0,  # Fixed risk
            rr_ratio=2.0,        # Fixed RR
            win_rate=wr,
            trades_per_day=tpd
        )
        
        avg_days = 30 - (result["failed_time_pct"] / 100 * 30)  # Rough estimate
        
        print(f"{name:<20} {result['pass_rate']:<8.1f} €{result['ev_per_attempt']:<9,.0f} "
              f"{avg_days:<10.1f} {result['failed_daily_dd_pct']:<10.1f}")


def main():
    print("="*80)
    print("PURE VARIANCE PLAY - PROP FIRM CHALLENGE OPTIMIZER")
    print("="*80)
    print("""
KEY INSIGHT: We don't need alpha to profit from prop firm challenges.

The structure is inherently exploitable:
- Fee: €540
- Potential payout: €10,000+
- Even 10% pass rate = +EV

The challenge is a CHEAP CALL OPTION on your variance.
    """)
    
    # Find optimal params
    results = find_optimal_params()
    
    # Compare crypto vs forex
    analyze_crypto_vs_forex()
    
    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    best = results[0]
    p = best["params"]
    
    print(f"""
OPTIMAL STRATEGY (50% win rate, no edge):
- Risk per trade: {p['risk']}%
- Risk:Reward: 1:{p['rr']}
- Trading frequency: {p['trades_per_day']} trades/day

Expected outcomes:
- Pass rate: {best['pass_rate']:.1f}%
- EV per attempt: €{best['ev_per_attempt']:,.0f}
- Expected attempts to pass: {best['expected_attempts_to_pass']:.1f}

CRITICAL RULES:
1. NEVER exceed daily DD (5%) - this is the #1 killer
2. Trade high-volatility instruments (faster resolution)
3. Don't overtrade - {p['trades_per_day']:.1f} quality setups/day is enough
4. Accept losses as cost of variance

Serial attempt strategy:
- Budget €2,000 for 3-4 attempts
- Expected profit after 4 attempts: €{(best['pass_rate']/100 * 4 * 10000 - 540*4):,.0f}
    """)
    
    print("✅ Results saved to variance_play_results.json")


if __name__ == "__main__":
    main()
