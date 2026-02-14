# 🌿 Curupira Research

Open-source quantitative trading strategies, backtests, and signal systems. Every strategy, every failure, every line of code.

**Blog:** [curupira.dev](https://curupira.dev)

## What's Here

### `/strategies/` — Signal Research
Original strategy implementations from scientific papers:

| Strategy | Paper | Status |
|----------|-------|--------|
| **ECVT** (Entropy Collapse Volatility Timing) | [Singha 2025, arXiv:2512.15720](https://arxiv.org/abs/2512.15720) | ✅ PF 1.44, +198 bps OOS |
| InfoTheo (Complexity-Gated Regime) | Shternshis & Marmi 2024 | ⚠️ Filter only, no standalone alpha |
| Biology (Flocking Contagion) | Kazemian et al. 2020 | ❌ Needs 30+ stock basket |
| Network (VG-TDA Regime) | Serafino et al. 2017 | ❌ O(n²), too expensive |

### `/backtests/` — Backtest Infrastructure
- `backtest_harness.py` — Unified 5-strategy harness (FVG, Williams %R, SMA/EMA, Range Breakout, Markov HMM)
- `fvg_tick_backtest.py` — Tick-level FVG backtester (42M ticks EURUSD)
- `rr_backtest.py` — Risk:Reward sweep across strategies
- `jump_tick_backtest.py` — Jump ratio tick verification
- `ecvt_fast.py` — Optimized ECVT implementation
- Signal profilers: `jump_ratio_profile.py`, `hurst_profile.py`

### `/signals/` — Live Signal System
- `signal_monitor.py` — Automated signal scanner (systemd timer, 30min)
- `signal_charts.py` — mplfinance JPEG chart generation
- `check_signals.py` — Quick signal checker

### `/llm-agent/` — LLM Trading Agent
Autonomous trading agent for Hyperliquid crypto perpetual futures.
- `llm_trading_agent.py` — Core agent with DeepSeek-style reasoning
- `onchain_collector.py` — Funding rates, OI, liquidation data
- `liq_watcher.py` — Liquidation cascade detector
- `command_server.py` — Command center dashboard
- Est. cost: $1.30/day

## The Graveyard

| Strategy | Timeframe | Result | Why It Died |
|----------|-----------|--------|-------------|
| FVG Magnetism | All TFs | ❌ DEAD | OHLC inflated 4×. Tick PF 1.04. [Details](https://curupira.dev/blog/fvg-magnetism-fair-value-gaps/) |
| Jump Fade | Forex 1H | ❌ DEAD | 50-51% WR at all thresholds. Zero predictive power. |
| Hurst Regime | Crypto 1H | ❌ DEAD | H>0.6 fires 100% of bars (mean=0.758). Not a filter. |
| ECVT | Equities | ❌ DEAD | 0-9 signals in 2 years. Forex-specific. |
| Jump+Trend Composite | Forex 1H | ❌ DEAD | Worse than either component alone (-3000 pips). |
| **ECVT** | **EURUSD 1H** | **✅ ALIVE** | **PF 1.44, +198 bps, 41% WR, walk-forward validated** |
| **Jump Trend** | **EURUSD 1H** | **✅ ALIVE** | **55% WR at 1:1, +293 pips. Decomposition win.** |

## Methodology

- **Walk-forward validation:** 6M train / 2M OOS / 2M step. No in-sample optimization.
- **Tick verification:** OHLC results are not trusted. All claims verified on 42M tick dataset.
- **Distribution profiling:** Signal distributions audited BEFORE backtesting. A signal that fires 100% has zero information.
- **Next-bar entry:** All Close-based signals enter at next bar's open. No look-ahead.
- **Honest reporting:** Most things fail. We publish the failures because the graveyard teaches more than the trophy case.

## Papers Referenced

- Singha (2025) — "Hidden Order in Trades Predicts the Size of Price Moves" — [arXiv:2512.15720](https://arxiv.org/abs/2512.15720)
- Shternshis & Marmi (2024) — Entropy-based predictability at ultra-high frequency
- Kazemian et al. (2020) — "Market of Stocks during Crisis Looks Like a Flock of Birds" (Entropy)
- Serafino et al. (2017) — Visibility graph validation for financial instability
- Montero (2009) — "Predator-Prey Model for Stock Market Fluctuations" — [arXiv:0810.4844](https://arxiv.org/abs/0810.4844)
- Lempel & Ziv (1976) — "On the Complexity of Finite Sequences"

## License

MIT. Use it, replicate it, improve it.

---

*The Curupira is a Brazilian forest spirit with backwards feet. Our research starts from the end — what would make this fail? — and walks toward the truth.*
