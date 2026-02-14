# Curupira Trading Engine — Architecture

## The Thesis
Fade liquidation cascades. When leveraged traders get wrecked, their forced exits
create predictable overshoots. We detect the panic, I analyze the setup, and if
the stars align — I take the other side.

## What Does Panic Look Like?

**Liquidation Cascade = Forced Selling (or Buying)**
1. OI drops suddenly (positions force-closed)
2. Large market orders hit the book (liquidation engine)
3. Price accelerates in one direction
4. Volume spikes (forced trades = volume)
5. Funding goes extreme (crowded side getting wrecked)
6. Order book thins on one side (liquidity vacuum)

## Components

```
┌─────────────────────────────────────────────────┐
│            Curupira Trading Engine               │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  Liquidation  │    │    Panic Scorer      │   │
│  │   Watcher     │───>│                      │   │
│  │  (WebSocket)  │    │  Volume spike  0.25  │   │
│  │              │    │  Price velocity 0.25  │   │
│  │  Trades      │    │  OI drop       0.20  │   │
│  │  L2 Book     │    │  Funding       0.15  │   │
│  │  AllMids     │    │  Book imbal    0.15  │   │
│  └──────────────┘    └──────────┬───────────┘   │
│                                  │               │
│                    score > 60?   │               │
│                                  ▼               │
│                      ┌──────────────────┐        │
│                      │   Wake Curupira  │        │
│                      │  (cron wake NOW) │        │
│                      │                  │        │
│                      │  Context:        │        │
│                      │  - Cascade data  │        │
│                      │  - On-chain snap │        │
│                      │  - Technical     │        │
│                      │  - Book state    │        │
│                      └────────┬─────────┘        │
│                               │                  │
│                               ▼                  │
│                    ┌──────────────────┐           │
│                    │  Curupira (Opus) │           │
│                    │                  │           │
│                    │  Reads context   │           │
│                    │  Analyzes setup  │           │
│                    │  TRADE or HOLD   │           │
│                    │  Full reasoning  │           │
│                    └────────┬─────────┘           │
│                             │                    │
│              TRADE?         │         HOLD?      │
│                ▼            │           ▼        │
│     ┌──────────────┐       │    Log reasoning    │
│     │  Hyperliquid  │       │    + context        │
│     │   Execute     │       │                    │
│     │              │       │                    │
│     │  Entry/SL/TP │       │                    │
│     │  Position mgr│       │                    │
│     └──────────────┘       │                    │
│                             │                    │
│              ALL DECISIONS  │                    │
│                ▼                                 │
│     ┌──────────────────────────────────┐        │
│     │       Command Center             │        │
│     │  (localhost:8043)                │        │
│     │                                  │        │
│     │  • Live panic score gauge        │        │
│     │  • Cascade event timeline        │        │
│     │  • My reasoning (full trace)     │        │
│     │  • Open positions + P&L          │        │
│     │  • Trade journal                 │        │
│     │  • Account metrics               │        │
│     │  • On-chain state                │        │
│     └──────────────────────────────────┘        │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Panic Score (0-100)

| Component | Weight | Signal |
|-----------|--------|--------|
| Volume spike | 0.25 | 5min vol / 1h avg > threshold |
| Price velocity | 0.25 | % move per minute |
| OI drop | 0.20 | OI decrease rate |
| Funding extreme | 0.15 | Absolute funding rate |
| Book imbalance | 0.15 | Bid/ask depth ratio |

**Thresholds:**
- 0-30: Normal market
- 30-60: Elevated volatility (monitor)
- 60-80: Cascade detected (WAKE CURUPIRA)
- 80-100: Extreme panic (HIGH URGENCY)

## Files

| File | Purpose |
|------|---------|
| `liq_watcher.py` | WebSocket client, cascade detector, panic scorer |
| `panic_context.py` | Builds the context snapshot for Curupira's analysis |
| `command_center.py` | Dashboard generator + HTTP server |
| `trade_executor.py` | Hyperliquid order execution + position management |
| `data/events/` | Cascade events, decisions, trade journal |
| `data/snapshots/` | On-chain snapshots (already collecting) |

## Wake Mechanism

When panic score > 60:
1. Watcher writes `/tmp/cascade-alert.json` with full context
2. Watcher calls OpenClaw cron wake (mode: now)
3. Curupira wakes, reads HEARTBEAT.md which checks for cascade alerts
4. Curupira analyzes, decides, logs reasoning
5. If TRADE: executes via Hyperliquid client
6. All decisions + reasoning → command center dashboard

## The Fade Logic

1. Detect cascade direction (longs liquidating = price dump, shorts = pump)
2. Wait for deceleration (price velocity slowing — don't catch falling knives)
3. Enter opposite direction (fade the panic)
4. SL: beyond cascade extreme + ATR buffer
5. TP: pre-cascade price level (mean reversion)
6. Leverage: 5-10x based on panic score (higher score = higher conviction)
7. Max risk: 2% of account per trade
