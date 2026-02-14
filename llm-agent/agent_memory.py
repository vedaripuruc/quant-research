#!/usr/bin/env python3
"""
Agent Memory / Trade Journal
==============================
SQLite-based trade journal that persists across sessions.
Agent reviews own history before making decisions.

"Last 3 times funding was this extreme, I did X and result was Y"
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TradeRecord:
    id: Optional[int]
    coin: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: str
    exit_price: Optional[float]
    exit_time: Optional[str]
    size: float
    leverage: float
    stop_loss: float
    take_profit: float
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    outcome: str  # OPEN, WIN, LOSS, BREAKEVEN
    entry_reason: str  # LLM's reasoning for entry
    exit_reason: Optional[str]  # LLM's reasoning for exit
    market_conditions: str  # snapshot of conditions at entry
    lessons: Optional[str]  # what the agent learned


class AgentMemory:
    """Persistent trade journal and memory for the trading agent."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), "trade_journal.db"
            )
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_price REAL,
                    exit_time TEXT,
                    size REAL NOT NULL,
                    leverage REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    outcome TEXT DEFAULT 'OPEN',
                    entry_reason TEXT,
                    exit_reason TEXT,
                    market_conditions TEXT,
                    lessons TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reasoning TEXT,
                    market_snapshot TEXT,
                    confidence REAL
                )
            """)
            conn.commit()

    def record_trade_open(self, coin: str, direction: str,
                          entry_price: float, size: float,
                          leverage: float, stop_loss: float,
                          take_profit: float, entry_reason: str,
                          market_conditions: str) -> int:
        """Record a new trade entry. Returns trade ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trades
                (coin, direction, entry_price, entry_time, size, leverage,
                 stop_loss, take_profit, entry_reason, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                coin, direction, entry_price,
                datetime.utcnow().isoformat(),
                size, leverage, stop_loss, take_profit,
                entry_reason, market_conditions,
            ))
            conn.commit()
            return cursor.lastrowid

    def record_trade_close(self, trade_id: int, exit_price: float,
                           pnl_usd: float, pnl_pct: float,
                           outcome: str, exit_reason: str,
                           lessons: Optional[str] = None):
        """Record a trade exit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades SET
                    exit_price = ?, exit_time = ?, pnl_usd = ?,
                    pnl_pct = ?, outcome = ?, exit_reason = ?, lessons = ?
                WHERE id = ?
            """, (
                exit_price, datetime.utcnow().isoformat(),
                pnl_usd, pnl_pct, outcome, exit_reason, lessons,
                trade_id,
            ))
            conn.commit()

    def record_decision(self, coin: str, action: str,
                        reasoning: str, market_snapshot: str,
                        confidence: float):
        """Record a decision (including HOLD decisions)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decisions
                (timestamp, coin, action, reasoning, market_snapshot, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                coin, action, reasoning, market_snapshot, confidence,
            ))
            conn.commit()

    def get_open_trades(self) -> List[Dict]:
        """Get all currently open trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE outcome = 'OPEN' ORDER BY entry_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get most recent closed trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE outcome != 'OPEN' "
                "ORDER BY exit_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trades_for_coin(self, coin: str, limit: int = 5) -> List[Dict]:
        """Get recent trades for a specific coin."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE coin = ? "
                "ORDER BY entry_time DESC LIMIT ?",
                (coin, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_performance_stats(self) -> Dict:
        """Calculate overall trading performance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            closed = conn.execute(
                "SELECT * FROM trades WHERE outcome != 'OPEN'"
            ).fetchall()

            if not closed:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0,
                    "best_trade": 0,
                    "worst_trade": 0,
                }

            trades = [dict(r) for r in closed]
            wins = [t for t in trades if t["outcome"] == "WIN"]
            losses = [t for t in trades if t["outcome"] == "LOSS"]

            total_wins = sum(t["pnl_usd"] or 0 for t in wins)
            total_losses = abs(sum(t["pnl_usd"] or 0 for t in losses))

            return {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "total_pnl": sum(t["pnl_usd"] or 0 for t in trades),
                "avg_win": total_wins / len(wins) if wins else 0,
                "avg_loss": -total_losses / len(losses) if losses else 0,
                "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
                "best_trade": max((t["pnl_usd"] or 0) for t in trades),
                "worst_trade": min((t["pnl_usd"] or 0) for t in trades),
            }

    def format_memory_context(self, coins: List[str]) -> str:
        """Format trade history as text for LLM context injection."""
        stats = self.get_performance_stats()
        recent = self.get_recent_trades(5)
        open_trades = self.get_open_trades()

        lines = [
            "=== Trade Journal ===",
            f"Total Trades: {stats['total_trades']} | "
            f"Win Rate: {stats['win_rate']:.0%} | "
            f"PF: {stats['profit_factor']:.2f} | "
            f"Total PnL: ${stats['total_pnl']:+,.2f}",
            "",
        ]

        if open_trades:
            lines.append("OPEN POSITIONS:")
            for t in open_trades:
                lines.append(
                    f"  {t['coin']} {t['direction']} @ ${t['entry_price']:,.2f} "
                    f"(SL: ${t['stop_loss']:,.2f}, TP: ${t['take_profit']:,.2f}) "
                    f"| Reason: {t['entry_reason'][:80]}"
                )
            lines.append("")

        if recent:
            lines.append("RECENT CLOSED TRADES:")
            for t in recent:
                pnl = t["pnl_usd"] or 0
                lines.append(
                    f"  {t['outcome']} {t['coin']} {t['direction']} "
                    f"${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                    f"PnL: ${pnl:+,.2f} | {t['exit_reason'][:60] if t['exit_reason'] else 'N/A'}"
                )
                if t.get("lessons"):
                    lines.append(f"    Lesson: {t['lessons'][:100]}")
            lines.append("")

        # Per-coin history for requested coins
        for coin in coins:
            coin_trades = self.get_trades_for_coin(coin, 3)
            if coin_trades:
                lines.append(f"History for {coin}:")
                for t in coin_trades:
                    pnl = t["pnl_usd"] or 0
                    lines.append(
                        f"  {t['outcome']} {t['direction']} "
                        f"PnL: ${pnl:+,.2f} | {t['entry_reason'][:60]}"
                    )

        return "\n".join(lines)
