#!/usr/bin/env python3
"""
LLM Trading Agent
==================
DeepSeek-style patient sniper + on-chain alpha.

Architecture:
- Single Claude agent (Sonnet) - no multi-agent overhead
- Technical: OHLCV + MACD + RSI (what DeepSeek uses)
- On-chain: funding rates, OI, liquidation zones (our unique edge)
- Strict risk rules baked into system prompt
- Low frequency: 1-3 trades/day max, hold 12-48 hours

Usage:
    python llm_trading_agent.py              # single analysis cycle
    python llm_trading_agent.py --loop       # continuous loop (30min intervals)
    python llm_trading_agent.py --dry-run    # analysis only, no trades
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import anthropic

from hyperliquid_client import HyperliquidClient
from onchain_feed import OnChainFeed
from technical_indicators import format_technical_context
from agent_memory import AgentMemory


# ── System Prompt (DeepSeek discipline + on-chain awareness) ──────

SYSTEM_PROMPT = """You are an autonomous crypto perpetual futures trading agent on Hyperliquid.
Your approach is modeled after the most successful AI traders in competition:
patient, disciplined, high-conviction, low-frequency.

## CORE RULES (NEVER VIOLATE)

1. PATIENCE: Target 1-3 trades per day MAXIMUM. Holding 12-48 hours is ideal.
   No trade is better than a mediocre trade. If conditions aren't clear, output HOLD.

2. RISK MANAGEMENT:
   - Never risk more than 2% of account value on a single trade
   - Every trade MUST have a stop-loss and take-profit
   - Maximum position size: 20% of account value
   - Maximum leverage: 15x (prefer 10x)
   - Minimum reward:risk ratio: 2:1

3. ENTRY DISCIPLINE:
   - "Buy on pullback, increase on breakout" (DeepSeek's rule)
   - Wait for confluence: at least 2 signals must align (technical + on-chain)
   - Never chase a move that's already extended

4. EXIT DISCIPLINE:
   - Hit stop-loss = close immediately, no hoping for recovery
   - Hit take-profit = close immediately, don't get greedy
   - If thesis is invalidated = close early, don't wait for SL

5. ON-CHAIN EDGE (what other AI traders miss):
   - Extreme funding rates signal crowded trades -> reversal risk
   - Rising OI + price rise = genuine momentum
   - Rising OI + price flat = positioning for breakout
   - Liquidation zones act as price magnets

## DECISION FRAMEWORK

For each analysis cycle, you receive:
- Technical indicators (MACD, RSI, EMA, ATR, Bollinger, volume)
- On-chain data (funding rates, open interest, liquidation zones)
- Your trade journal (past trades, performance stats)
- Current positions and account state

You must respond with a JSON object:

```json
{
  "action": "HOLD" | "OPEN_LONG" | "OPEN_SHORT" | "CLOSE",
  "coin": "BTC",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence explanation of your decision",
  "entry_price": null or float,
  "stop_loss": null or float,
  "take_profit": null or float,
  "size_pct": null or float (0.01 to 0.20 = 1-20% of account),
  "leverage": null or int (1-15),
  "close_trade_id": null or int (for CLOSE actions)
}
```

IMPORTANT: Only output the JSON object. No markdown, no extra text.

## WHAT MAKES A GOOD TRADE

HIGH CONVICTION SETUP (confidence > 0.7):
- RSI oversold/overbought reversal + extreme funding rate in opposite direction
- MACD bullish crossover + rising OI + price at Bollinger lower band
- Price approaching liquidation zone cluster + funding rate extreme

MEDIUM CONVICTION SETUP (confidence 0.5-0.7):
- Technical signal alignment without strong on-chain confirmation
- Only take these if win rate on similar past trades is > 60%

LOW CONVICTION (confidence < 0.5):
- Output HOLD. Always.
"""


class TradingAgent:
    """DeepSeek-style patient sniper trading agent."""

    WATCHED_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LINK"]
    SCAN_INTERVAL_SECONDS = 1800  # 30 minutes

    def __init__(self, private_key: str, testnet: bool = True,
                 dry_run: bool = False):
        self.dry_run = dry_run
        self.client = HyperliquidClient(private_key, testnet=testnet)
        self.onchain = OnChainFeed(testnet=testnet)
        self.memory = AgentMemory()
        self.claude = anthropic.Anthropic()

        # Track today's trade count
        self._today = datetime.utcnow().date()
        self._trades_today = 0

    def run_cycle(self) -> List[Dict]:
        """Run one analysis cycle across all watched coins."""
        # Reset daily counter
        today = datetime.utcnow().date()
        if today != self._today:
            self._today = today
            self._trades_today = 0

        print(f"\n{'='*60}")
        print(f"Analysis Cycle: {datetime.utcnow().isoformat()}")
        print(f"Trades today: {self._trades_today}/3")
        print(f"{'='*60}")

        # Check open positions for exit conditions first
        self._check_open_positions()

        # Skip if we've already made 3 trades today
        if self._trades_today >= 3:
            print("Daily trade limit reached (3/3). Holding.")
            return []

        # Gather context for all coins
        decisions = []
        for coin in self.WATCHED_COINS:
            try:
                decision = self._analyze_coin(coin)
                if decision:
                    decisions.append(decision)
            except Exception as e:
                print(f"Error analyzing {coin}: {e}")

        return decisions

    def _analyze_coin(self, coin: str) -> Optional[Dict]:
        """Analyze a single coin and return agent's decision."""
        print(f"\n--- Analyzing {coin} ---")

        # 1. Get technical data
        df = self.client.get_candles(coin, interval="1h", lookback_hours=100)
        if df.empty or len(df) < 30:
            print(f"  Insufficient data for {coin}")
            return None

        technical_ctx = format_technical_context(df, coin)

        # 2. Get on-chain data
        onchain_ctx = self.onchain.format_onchain_context([coin])

        # 3. Get account state
        account_ctx = self.client.format_account_context()

        # 4. Get trade memory
        memory_ctx = self.memory.format_memory_context([coin])

        # 5. Build the prompt
        user_prompt = f"""Current time: {datetime.utcnow().isoformat()}
Trades made today: {self._trades_today}/3

{technical_ctx}

{onchain_ctx}

{account_ctx}

{memory_ctx}

Analyze {coin} and provide your trading decision as JSON."""

        # 6. Call Claude
        print(f"  Calling Claude for {coin} analysis...")
        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text.strip()

            # Parse JSON from response (handle markdown code blocks)
            json_text = raw_text
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
            json_text = json_text.strip()

            decision = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"  Failed to parse Claude response: {e}")
            print(f"  Raw: {raw_text[:200]}")
            return None
        except Exception as e:
            print(f"  Claude API error: {e}")
            return None

        # 7. Log the decision
        action = decision.get("action", "HOLD")
        confidence = decision.get("confidence", 0)
        reasoning = decision.get("reasoning", "")

        print(f"  Decision: {action} (confidence: {confidence:.0%})")
        print(f"  Reasoning: {reasoning}")

        # Record every decision (including HOLDs)
        self.memory.record_decision(
            coin=coin,
            action=action,
            reasoning=reasoning,
            market_snapshot=f"Technical:\n{technical_ctx[:500]}\nOnchain:\n{onchain_ctx[:500]}",
            confidence=confidence,
        )

        # 8. Execute if not dry run
        if action != "HOLD" and not self.dry_run:
            self._execute_decision(coin, decision, technical_ctx, onchain_ctx)

        return decision

    def _execute_decision(self, coin: str, decision: Dict,
                          technical_ctx: str, onchain_ctx: str):
        """Execute a trading decision."""
        action = decision["action"]
        confidence = decision.get("confidence", 0)

        # Safety: never trade on low confidence
        if confidence < 0.5:
            print(f"  Confidence too low ({confidence:.0%}), skipping execution")
            return

        if action in ("OPEN_LONG", "OPEN_SHORT"):
            self._open_position(coin, decision, technical_ctx, onchain_ctx)
        elif action == "CLOSE":
            self._close_position(coin, decision)

    def _open_position(self, coin: str, decision: Dict,
                       technical_ctx: str, onchain_ctx: str):
        """Open a new position."""
        is_buy = decision["action"] == "OPEN_LONG"
        direction = "LONG" if is_buy else "SHORT"
        leverage = min(decision.get("leverage", 10), 15)  # cap at 15x
        size_pct = min(decision.get("size_pct", 0.05), 0.20)  # cap at 20%
        stop_loss = decision.get("stop_loss")
        take_profit = decision.get("take_profit")

        if not stop_loss or not take_profit:
            print("  Missing SL or TP, skipping")
            return

        # Calculate position size
        account_value = self.client.get_account_value()
        if account_value <= 0:
            print("  No account value, skipping")
            return

        notional = account_value * size_pct
        mid = self.client.get_mid_price(coin)
        if not mid:
            print(f"  Cannot get price for {coin}")
            return

        size = (notional * leverage) / mid

        # Risk check: ensure risk < 2% of account
        sl_distance_pct = abs(mid - stop_loss) / mid
        risk_amount = notional * sl_distance_pct * leverage
        max_risk = account_value * 0.02
        if risk_amount > max_risk:
            # Scale down size to stay within risk limits
            scale = max_risk / risk_amount
            size *= scale
            notional *= scale
            print(f"  Scaled position down to stay within 2% risk limit")

        # Set leverage
        self.client.set_leverage(coin, leverage)

        # Execute
        print(f"  Opening {direction} {coin}: size={size:.4f}, leverage={leverage}x")
        result = self.client.market_open(coin, is_buy, size)

        if result.success:
            self._trades_today += 1
            trade_id = self.memory.record_trade_open(
                coin=coin,
                direction=direction,
                entry_price=mid,
                size=size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_reason=decision.get("reasoning", ""),
                market_conditions=f"{technical_ctx[:300]}\n{onchain_ctx[:300]}",
            )
            print(f"  SUCCESS: Trade #{trade_id} opened. Order ID: {result.order_id}")
        else:
            print(f"  FAILED: {result.error}")

    def _close_position(self, coin: str, decision: Dict):
        """Close an existing position."""
        pos = self.client.get_position(coin)
        if not pos:
            print(f"  No position to close for {coin}")
            return

        print(f"  Closing {coin} position...")
        result = self.client.market_close(coin)

        if result.success:
            # Find the open trade in memory
            open_trades = self.memory.get_open_trades()
            matching = [t for t in open_trades if t["coin"] == coin]
            if matching:
                trade = matching[0]
                exit_price = self.client.get_mid_price(coin) or pos.entry_price
                direction_mult = 1 if trade["direction"] == "LONG" else -1
                pnl_pct = direction_mult * (exit_price / trade["entry_price"] - 1) * trade["leverage"]
                pnl_usd = pnl_pct * (trade["size"] * trade["entry_price"] / trade["leverage"])

                outcome = "WIN" if pnl_usd > 0 else "LOSS"
                self.memory.record_trade_close(
                    trade_id=trade["id"],
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                    exit_reason=decision.get("reasoning", "Manual close"),
                )
                print(f"  Closed: {outcome} ${pnl_usd:+,.2f} ({pnl_pct:+.1%})")
        else:
            print(f"  Failed to close: {result.error}")

    def _check_open_positions(self):
        """Check open positions for SL/TP hits."""
        open_trades = self.memory.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            coin = trade["coin"]
            mid = self.client.get_mid_price(coin)
            if not mid:
                continue

            direction = trade["direction"]
            sl = trade["stop_loss"]
            tp = trade["take_profit"]
            hit = None

            if direction == "LONG":
                if mid <= sl:
                    hit = "stop_loss"
                elif mid >= tp:
                    hit = "take_profit"
            else:  # SHORT
                if mid >= sl:
                    hit = "stop_loss"
                elif mid <= tp:
                    hit = "take_profit"

            if hit:
                print(f"  {coin}: {hit.upper()} hit at ${mid:,.2f}")
                result = self.client.market_close(coin)
                if result.success:
                    direction_mult = 1 if direction == "LONG" else -1
                    pnl_pct = direction_mult * (mid / trade["entry_price"] - 1) * trade["leverage"]
                    pnl_usd = pnl_pct * (trade["size"] * trade["entry_price"] / trade["leverage"])
                    outcome = "WIN" if hit == "take_profit" else "LOSS"

                    self.memory.record_trade_close(
                        trade_id=trade["id"],
                        exit_price=mid,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        outcome=outcome,
                        exit_reason=f"{hit} hit at ${mid:,.2f}",
                    )
                    print(f"    {outcome}: ${pnl_usd:+,.2f}")

    def run_loop(self):
        """Continuous loop - analyze every 30 minutes."""
        print("Starting continuous trading loop...")
        print(f"  Coins: {', '.join(self.WATCHED_COINS)}")
        print(f"  Interval: {self.SCAN_INTERVAL_SECONDS}s")
        print(f"  Dry run: {self.dry_run}")
        print(f"  Network: {'TESTNET' if self.client.testnet else 'MAINNET'}")
        print()

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                print("\nStopping agent...")
                break
            except Exception as e:
                print(f"\nCycle error: {e}")

            print(f"\nNext scan in {self.SCAN_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(self.SCAN_INTERVAL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="LLM Trading Agent")
    parser.add_argument("--loop", action="store_true",
                        help="Run in continuous loop mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analysis only, no trades executed")
    parser.add_argument("--mainnet", action="store_true",
                        help="Use mainnet (default: testnet)")
    parser.add_argument("--coins", type=str, default=None,
                        help="Comma-separated coins to watch (e.g. BTC,ETH,SOL)")
    args = parser.parse_args()

    # Load private key from environment
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY")
    if not private_key:
        # Try config file
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                private_key = config.get("secret_key")

    if not private_key:
        print("ERROR: No private key found.")
        print("Set HYPERLIQUID_PRIVATE_KEY env var or create config.json with:")
        print('  {"secret_key": "0x...", "account_address": "0x..."}')
        sys.exit(1)

    # Check for Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Export your Anthropic API key: export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    testnet = not args.mainnet
    agent = TradingAgent(
        private_key=private_key,
        testnet=testnet,
        dry_run=args.dry_run,
    )

    if args.coins:
        agent.WATCHED_COINS = [c.strip().upper() for c in args.coins.split(",")]

    if args.loop:
        agent.run_loop()
    else:
        agent.run_cycle()


if __name__ == "__main__":
    main()
