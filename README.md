# Quant Research

Open-source quantitative trading research. Strategies from scientific papers, tick-level backtests, and an autonomous crypto trading agent. Most strategies failed — that's the point.

**Blog:** [curupira.dev](https://curupira.dev)

## Structure

### `/strategies/` — Signal Implementations

One file per strategy, each derived from a published paper. Status reflects honest out-of-sample results.

| Strategy | Source Paper | Result |
|----------|-------------|--------|
| **ECVT** — Entropy Collapse Volatility Timing | [Singha 2025](https://arxiv.org/abs/2512.15720) | ☠️ Dead on equities/cross-pairs. Forex-specific artifact. |
| **InfoTheo** — Complexity-Gated Regime | Shternshis & Marmi 2024 | ☠️ Works as filter only, no standalone alpha. |
| **Biology** — Flocking Contagion | Kazemian et al. 2020 | ☠️ Needs 30+ stock basket. Wrong asset class for forex. |
| **Network** — VG-TDA Regime Detection | Serafino et al. 2017 | ☠️ O(n²), too expensive to run at scale. |
| **Physics** — Predator-Prey Dynamics | [Montero 2009](https://arxiv.org/abs/0810.4844) | ☠️ Signal exists but not tradable standalone. |

### `/backtests/` — Infrastructure

- `backtest_harness.py` — Unified multi-strategy harness (FVG, Williams %R, SMA/EMA, Range Breakout, Markov HMM)
- `fvg_tick_backtest.py` — Tick-level FVG backtester (tested on 42M ticks EURUSD)
- `walk_forward.py` — Walk-forward validation engine (6M train / 2M OOS)
- `aggregate_ticks.py` — Tick data aggregation to OHLCV bars

### `/signals/` — Live Signal System

- `signal_monitor.py` — Automated signal scanner (designed for systemd timer)
- `signal_charts.py` — mplfinance chart generation
- `check_signals.py` — Quick signal status check

### `/llm-agent/` — Autonomous Crypto Trading Agent

An LLM-powered trading agent for Hyperliquid perpetual futures. Collects on-chain data, detects liquidation cascades, and reasons about trades using DeepSeek-style chain-of-thought.

- `llm_trading_agent.py` — Core agent with structured reasoning
- `onchain_collector.py` — Funding rates, open interest, liquidation snapshots
- `liq_watcher.py` — Liquidation cascade detector with rolling statistics
- `hyperliquid_client.py` — Exchange API client
- `command_server.py` — Dashboard and command interface
- `agent_memory.py` — Persistent memory across sessions
- Est. running cost: ~$1.30/day

## Methodology

- **Walk-forward validation** — 6M train / 2M OOS / 2M step. No in-sample optimization.
- **Tick verification** — OHLC backtests are not trusted. Claims verified on tick data.
- **Distribution audit** — Signal firing rates checked before backtesting. 100% fire rate = zero information.
- **Next-bar entry** — All signals enter at next bar's open. No look-ahead bias.
- **Honest reporting** — Most strategies fail. The graveyard is the curriculum.

Read more: [We Tested Every Strategy We Could Find. Most Failed.](https://curupira.dev/blog/31-strategies-tested-4-survived)

## Papers

- Singha (2025) — "Hidden Order in Trades Predicts the Size of Price Moves" — [arXiv:2512.15720](https://arxiv.org/abs/2512.15720)
- Shternshis & Marmi (2024) — Entropy-based predictability at ultra-high frequency
- Kazemian et al. (2020) — "Market of Stocks during Crisis Looks Like a Flock of Birds"
- Serafino et al. (2017) — Visibility graph validation for financial instability
- Montero (2009) — "Predator-Prey Model for Stock Market Fluctuations" — [arXiv:0810.4844](https://arxiv.org/abs/0810.4844)
- Lempel & Ziv (1976) — "On the Complexity of Finite Sequences"

## License

MIT
