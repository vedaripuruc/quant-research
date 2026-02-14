#!/usr/bin/env python3
"""Unified walk-forward backtest harness for bang-indicators strategies."""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import DateOffset

from backend.app import breakout, crossover, fvg, markov, williamsR

ASSETS = ["EURUSD=X", "GBPUSD=X", "SPY", "BTC-USD", "GC=F"]
TIMEFRAMES = ["1h", "1d"]
START_TS = pd.Timestamp("2024-02-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-01", tz="UTC")
TRAIN_MONTHS = 6
TEST_MONTHS = 2
STEP_MONTHS = 2
ANNUALIZATION = {"1h": 252 * 24, "1d": 252}
RESULTS_DIR = Path("results")


@dataclass
class WalkForwardWindow:
    """A single walk-forward split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class WindowRun:
    """Normalized output for one strategy on one walk-forward window."""

    equity_curve: pd.Series
    trade_samples: list[tuple[pd.Timestamp, float]]


@dataclass
class StrategySpec:
    """Strategy configuration and runner callback."""

    name: str
    runner: Callable[[pd.DataFrame, pd.DataFrame], WindowRun]
    intraday_only: bool = False


def configure_runtime() -> None:
    """Configure logging, caches, and warning filters for deterministic runs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    for noisy_logger in ["backend.app.breakout", "backend.app.crossover", "yfinance", "matplotlib"]:
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("YFINANCE_CACHE_DIR", "/tmp/yfinance-cache")
    Path(os.environ["YFINANCE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location("/tmp/yfinance-tz")

    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output into a canonical OHLCV shape."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    else:
        out.index = pd.to_datetime(out.index, errors="coerce", utc=True)

    out = out[~out.index.isna()].sort_index()
    out = out.loc[(out.index >= START_TS) & (out.index < END_TS)]
    out["Date"] = out.index
    return out


def download_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Download historical data for one asset/timeframe combination."""
    interval = timeframe

    try:
        raw = yf.download(
            asset,
            start=START_TS.strftime("%Y-%m-%d"),
            end=END_TS.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance")

    prepared = _prepare_ohlcv(raw)
    if prepared.empty:
        raise RuntimeError("Data became empty after normalization")
    return prepared


def make_windows(df: pd.DataFrame) -> list[WalkForwardWindow]:
    """Create rolling 6M train / 2M test walk-forward windows."""
    windows: list[WalkForwardWindow] = []
    cursor = START_TS

    while True:
        train_end = cursor + DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + DateOffset(months=TEST_MONTHS)

        if test_end > END_TS:
            break

        train_df = df.loc[(df.index >= cursor) & (df.index < train_end)]
        test_df = df.loc[(df.index >= train_end) & (df.index < test_end)]

        if not train_df.empty and not test_df.empty:
            windows.append(
                WalkForwardWindow(
                    train_start=cursor,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    train_df=train_df.copy(),
                    test_df=test_df.copy(),
                )
            )

        cursor = cursor + DateOffset(months=STEP_MONTHS)

    return windows


def to_strategy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a strategy-friendly frame with a Date column and positional index."""
    out = df.copy()
    if "Date" not in out.columns:
        out["Date"] = out.index
    return out.reset_index(drop=True)


def safe_float(value: Any) -> float | None:
    """Safely cast to finite float."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(as_float):
        return None
    return as_float


def normalize_trades(raw: Any) -> pd.DataFrame:
    """Normalize raw strategy output into a trade DataFrame if possible."""
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, tuple) and raw and isinstance(raw[0], pd.DataFrame):
        return raw[0].copy()
    if isinstance(raw, dict):
        trades = raw.get("trades")
        if isinstance(trades, pd.DataFrame):
            return trades.copy()
        if isinstance(trades, list):
            return pd.DataFrame(trades)
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    return pd.DataFrame()


def parse_trade_time(value: Any, index_ref: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Parse trade timestamp from either integer bar index or datetime-like value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
        numeric = safe_float(value)
        if numeric is not None:
            idx = int(numeric)
            if 0 <= idx < len(index_ref):
                return pd.Timestamp(index_ref[idx]).tz_convert("UTC")

    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).tz_convert("UTC")


def infer_direction(row: pd.Series) -> str:
    """Infer long/short direction from strategy trade row."""
    candidates = [
        row.get("Trade_Direction"),
        row.get("Direction"),
        row.get("Side"),
        row.get("Type"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        if text in {"long", "bullish", "buy"}:
            return "long"
        if text in {"short", "bearish", "sell"}:
            return "short"

    return "long"


def align_to_index(ts: pd.Timestamp, index_ref: pd.DatetimeIndex) -> pd.Timestamp:
    """Clamp/align a timestamp to the nearest existing bar in index_ref."""
    if ts <= index_ref[0]:
        return pd.Timestamp(index_ref[0])
    if ts >= index_ref[-1]:
        return pd.Timestamp(index_ref[-1])

    pos = index_ref.searchsorted(ts, side="left")
    if pos >= len(index_ref):
        pos = len(index_ref) - 1
    return pd.Timestamp(index_ref[pos])


def extract_trade_samples(
    trades: pd.DataFrame,
    index_ref: pd.DatetimeIndex,
    filter_start: pd.Timestamp,
    filter_end: pd.Timestamp,
    filter_on_entry: bool,
) -> list[tuple[pd.Timestamp, float]]:
    """Extract (exit_timestamp, trade_return) samples from heterogeneous trade tables."""
    samples: list[tuple[pd.Timestamp, float]] = []

    if trades.empty or index_ref.empty:
        return samples

    for _, row in trades.iterrows():
        entry_price = safe_float(row.get("Entry_Price"))
        exit_price = safe_float(row.get("Exit_Price"))
        if entry_price is None or exit_price is None or entry_price == 0:
            continue

        direction = infer_direction(row)
        if direction == "short":
            trade_return = (entry_price - exit_price) / entry_price
        else:
            trade_return = (exit_price - entry_price) / entry_price

        entry_time = parse_trade_time(row.get("Entry_Time"), index_ref)
        exit_time = parse_trade_time(row.get("Exit_Time"), index_ref)

        if exit_time is None:
            exit_time = index_ref[-1]
        if entry_time is None:
            entry_time = exit_time

        candidate = entry_time if filter_on_entry else exit_time
        if not (filter_start <= candidate < filter_end):
            continue

        aligned_exit = align_to_index(exit_time, index_ref)
        samples.append((aligned_exit, trade_return))

    samples.sort(key=lambda item: item[0])
    return samples


def equity_from_trade_samples(
    index_ref: pd.DatetimeIndex,
    samples: list[tuple[pd.Timestamp, float]],
) -> pd.Series:
    """Build stepwise equity curve over a test window from trade returns."""
    if index_ref.empty:
        return pd.Series(dtype=float)

    events: dict[pd.Timestamp, list[float]] = {}
    for ts, trade_return in samples:
        events.setdefault(ts, []).append(trade_return)

    equity = 1.0
    values: list[float] = []

    for ts in index_ref:
        for trade_return in events.get(pd.Timestamp(ts), []):
            equity *= 1.0 + trade_return
        values.append(equity)

    return pd.Series(values, index=index_ref, dtype=float)


def run_fvg_window(_: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run FVG strategy on one out-of-sample window."""
    strategy_df = to_strategy_df(test_df)
    raw = fvg.backtest_fvg_strategy(strategy_df.copy())
    trades = normalize_trades(raw)

    test_start = pd.Timestamp(test_df.index[0])
    test_end = pd.Timestamp(test_df.index[-1]) + pd.Timedelta(microseconds=1)
    samples = extract_trade_samples(trades, test_df.index, test_start, test_end, filter_on_entry=False)
    equity_curve = equity_from_trade_samples(test_df.index, samples)
    return WindowRun(equity_curve=equity_curve, trade_samples=samples)


def run_williams_window(_: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run Williams %R strategy on one out-of-sample window."""
    strategy_df = to_strategy_df(test_df)
    enriched = williamsR.calculate_williams_r(strategy_df.copy())
    raw = williamsR.backtest_williams_r_strategy(enriched.copy())
    trades = normalize_trades(raw)

    test_start = pd.Timestamp(test_df.index[0])
    test_end = pd.Timestamp(test_df.index[-1]) + pd.Timedelta(microseconds=1)
    samples = extract_trade_samples(trades, test_df.index, test_start, test_end, filter_on_entry=False)
    equity_curve = equity_from_trade_samples(test_df.index, samples)
    return WindowRun(equity_curve=equity_curve, trade_samples=samples)


def run_crossover_window(_: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run SMA/EMA crossover strategy on one out-of-sample window."""
    strategy_df = to_strategy_df(test_df)
    enriched = crossover.calculate_sma_ema(strategy_df.copy())
    raw = crossover.backtest_sma_ema_strategy(enriched.copy())
    trades = normalize_trades(raw)

    test_start = pd.Timestamp(test_df.index[0])
    test_end = pd.Timestamp(test_df.index[-1]) + pd.Timedelta(microseconds=1)
    samples = extract_trade_samples(trades, test_df.index, test_start, test_end, filter_on_entry=False)
    equity_curve = equity_from_trade_samples(test_df.index, samples)
    return WindowRun(equity_curve=equity_curve, trade_samples=samples)


def run_breakout_window(_: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run range S/R breakout strategy on one out-of-sample window."""
    strategy_df = to_strategy_df(test_df)
    ranged = breakout.calculate_daily_range(strategy_df.copy())
    raw = breakout.backtest_range_support_resistance_strategy(ranged.copy())
    trades = normalize_trades(raw)

    test_start = pd.Timestamp(test_df.index[0])
    test_end = pd.Timestamp(test_df.index[-1]) + pd.Timedelta(microseconds=1)
    samples = extract_trade_samples(trades, test_df.index, test_start, test_end, filter_on_entry=False)
    equity_curve = equity_from_trade_samples(test_df.index, samples)
    return WindowRun(equity_curve=equity_curve, trade_samples=samples)


def run_markov_hmm_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run Markov HMM strategy using train+test context, scoring only OOS trades."""
    combined = pd.concat([train_df, test_df]).sort_index()
    strategy_df = to_strategy_df(combined)

    original_plotter = markov.plot_trades_markov
    markov.plot_trades_markov = lambda *_args, **_kwargs: ""
    try:
        raw = markov.run_markov_strategy(strategy_df.copy())
    finally:
        markov.plot_trades_markov = original_plotter

    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(str(raw["error"]))

    trades = normalize_trades(raw)
    test_start = pd.Timestamp(test_df.index[0])
    test_end = pd.Timestamp(test_df.index[-1]) + pd.Timedelta(microseconds=1)
    samples = extract_trade_samples(trades, combined.index, test_start, test_end, filter_on_entry=True)
    equity_curve = equity_from_trade_samples(test_df.index, samples)
    return WindowRun(equity_curve=equity_curve, trade_samples=samples)


def markov_detailed_to_window(
    detailed: dict[str, Any],
    combined_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
) -> WindowRun:
    """Convert run_markov_strategy_detailed output into normalized OOS equity/trades."""
    cumulative_raw = pd.to_numeric(pd.Series(detailed.get("cumulative_returns", [])), errors="coerce")
    if cumulative_raw.empty:
        test_equity = pd.Series(1.0, index=test_index, dtype=float)
    else:
        if len(cumulative_raw) < len(combined_index):
            cumulative_raw = cumulative_raw.reindex(range(len(combined_index)))
        cumulative_raw = cumulative_raw.iloc[: len(combined_index)]

        full_equity = pd.Series(cumulative_raw.to_numpy(dtype=float), index=combined_index)
        full_equity = full_equity.replace([np.inf, -np.inf], np.nan).ffill()

        if full_equity.dropna().empty:
            full_equity = pd.Series(1.0, index=combined_index, dtype=float)

        test_equity = full_equity.reindex(test_index).ffill().replace([np.inf, -np.inf], np.nan)
        if test_equity.dropna().empty:
            test_equity = pd.Series(1.0, index=test_index, dtype=float)
        else:
            first_value = float(test_equity.dropna().iloc[0])
            if not np.isfinite(first_value) or first_value == 0:
                first_value = 1.0
            test_equity = (test_equity / first_value).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    signals_raw = pd.to_numeric(pd.Series(detailed.get("signals", [])), errors="coerce")
    if signals_raw.empty:
        return WindowRun(equity_curve=test_equity, trade_samples=[])

    if len(signals_raw) < len(combined_index):
        signals_raw = signals_raw.reindex(range(len(combined_index)), fill_value=0.0)
    signals_raw = signals_raw.iloc[: len(combined_index)]

    full_signals = pd.Series(signals_raw.to_numpy(dtype=float), index=combined_index).fillna(0.0)
    test_signals = full_signals.reindex(test_index).fillna(0.0)

    trade_samples: list[tuple[pd.Timestamp, float]] = []
    segment_start: pd.Timestamp | None = None
    segment_side: float = 0.0
    previous_ts: pd.Timestamp | None = None

    for ts, signal in test_signals.items():
        signal_value = float(signal)

        if segment_start is None and signal_value != 0.0:
            segment_start = pd.Timestamp(ts)
            segment_side = signal_value
        elif segment_start is not None:
            side_changed = signal_value == 0.0 or np.sign(signal_value) != np.sign(segment_side)
            if side_changed:
                segment_end = previous_ts if previous_ts is not None else pd.Timestamp(ts)
                start_equity = safe_float(test_equity.loc[segment_start])
                end_equity = safe_float(test_equity.loc[segment_end])
                if start_equity and end_equity is not None and start_equity != 0:
                    trade_samples.append((segment_end, (end_equity / start_equity) - 1.0))

                if signal_value == 0.0:
                    segment_start = None
                    segment_side = 0.0
                else:
                    segment_start = pd.Timestamp(ts)
                    segment_side = signal_value

        previous_ts = pd.Timestamp(ts)

    if segment_start is not None:
        segment_end = pd.Timestamp(test_index[-1])
        start_equity = safe_float(test_equity.loc[segment_start])
        end_equity = safe_float(test_equity.loc[segment_end])
        if start_equity and end_equity is not None and start_equity != 0:
            trade_samples.append((segment_end, (end_equity / start_equity) - 1.0))

    trade_samples.sort(key=lambda item: item[0])
    return WindowRun(equity_curve=test_equity.astype(float), trade_samples=trade_samples)


def run_markov_detailed_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> WindowRun:
    """Run detailed Markov variant using train+test context, scoring only OOS slice."""
    combined = pd.concat([train_df, test_df]).sort_index()
    strategy_df = to_strategy_df(combined)

    raw = markov.run_markov_strategy_detailed(strategy_df.copy())
    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(str(raw["error"]))
    if not isinstance(raw, dict):
        raise RuntimeError("run_markov_strategy_detailed returned unexpected type")

    return markov_detailed_to_window(raw, combined.index, test_df.index)


def concat_window_curves(window_runs: list[WindowRun]) -> pd.Series:
    """Chain window-level normalized curves into a single cumulative OOS curve."""
    curves: list[pd.Series] = []
    capital = 1.0

    for run in window_runs:
        curve = run.equity_curve.dropna().astype(float)
        if curve.empty:
            continue

        scaled = curve * capital
        capital = float(scaled.iloc[-1])

        if curves:
            previous_last_ts = curves[-1].index[-1]
            scaled = scaled.loc[scaled.index > previous_last_ts]

        if not scaled.empty:
            curves.append(scaled)

    if not curves:
        return pd.Series(dtype=float)

    merged = pd.concat(curves).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def buy_and_hold_return(windows: list[WalkForwardWindow]) -> float:
    """Compute buy-and-hold return over concatenated OOS windows."""
    close_segments = [window.test_df["Close"] for window in windows if not window.test_df.empty]
    if not close_segments:
        return 0.0

    close_series = pd.concat(close_segments).sort_index()
    close_series = close_series[~close_series.index.duplicated(keep="first")]
    if close_series.empty:
        return 0.0

    first = safe_float(close_series.iloc[0])
    last = safe_float(close_series.iloc[-1])
    if first is None or last is None or first == 0:
        return 0.0
    return ((last / first) - 1.0) * 100.0


def compute_metrics(
    equity_curve: pd.Series,
    trade_samples: list[tuple[pd.Timestamp, float]],
    timeframe: str,
    buy_hold_pct: float,
) -> dict[str, float | None]:
    """Compute standard performance metrics from normalized equity and trade samples."""
    if equity_curve.empty:
        return {
            "Total Return (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Profit Factor": 0.0,
            "Win Rate (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Number of Trades": 0.0,
            "Average Win (%)": 0.0,
            "Average Loss (%)": 0.0,
            "Buy and Hold Return (%)": buy_hold_pct,
        }

    total_return_pct = (float(equity_curve.iloc[-1]) - 1.0) * 100.0

    bar_returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe_ratio = 0.0
    if len(bar_returns) > 1:
        std = float(bar_returns.std(ddof=0))
        if std > 0:
            sharpe_ratio = float(np.sqrt(ANNUALIZATION[timeframe]) * bar_returns.mean() / std)

    drawdown = (equity_curve / equity_curve.cummax()) - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    trade_returns = [sample[1] for sample in trade_samples]
    wins = [value for value in trade_returns if value > 0]
    losses = [value for value in trade_returns if value < 0]

    number_of_trades = float(len(trade_returns))
    win_rate_pct = (len(wins) / len(trade_returns) * 100.0) if trade_returns else 0.0

    avg_win_pct = (float(np.mean(wins)) * 100.0) if wins else 0.0
    avg_loss_pct = (float(np.mean(losses)) * 100.0) if losses else 0.0

    sum_wins = float(np.sum(wins)) if wins else 0.0
    sum_losses = abs(float(np.sum(losses))) if losses else 0.0
    if sum_losses == 0:
        profit_factor = float("inf") if sum_wins > 0 else 0.0
    else:
        profit_factor = sum_wins / sum_losses

    return {
        "Total Return (%)": total_return_pct,
        "Sharpe Ratio": sharpe_ratio,
        "Profit Factor": profit_factor,
        "Win Rate (%)": win_rate_pct,
        "Max Drawdown (%)": max_drawdown_pct,
        "Number of Trades": number_of_trades,
        "Average Win (%)": avg_win_pct,
        "Average Loss (%)": avg_loss_pct,
        "Buy and Hold Return (%)": buy_hold_pct,
    }


def clean_number(value: Any, precision: int = 4) -> float | None:
    """Round finite numeric values for stable JSON output."""
    numeric = safe_float(value)
    if numeric is None:
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, precision)


def format_value(value: Any, digits: int = 2) -> str:
    """Human-friendly formatter for stdout/markdown tables."""
    numeric = safe_float(value)
    if numeric is None or not np.isfinite(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def build_markdown(summary_rows: list[dict[str, Any]]) -> str:
    """Create markdown summary table for RESULTS.md."""
    lines = [
        "# Backtest Summary",
        "",
        "| Strategy | Asset | TF | Status | Total Return % | Sharpe | Profit Factor | Win Rate % | Max DD % | Trades | Avg Win % | Avg Loss % | Buy&Hold % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        metrics = row.get("metrics", {})
        lines.append(
            "| {strategy} | {asset} | {tf} | {status} | {total} | {sharpe} | {pf} | {win_rate} | {max_dd} | {trades} | {avg_win} | {avg_loss} | {buy_hold} |".format(
                strategy=row["strategy"],
                asset=row["asset"],
                tf=row["timeframe"],
                status=row["status"],
                total=format_value(metrics.get("Total Return (%)")),
                sharpe=format_value(metrics.get("Sharpe Ratio")),
                pf=format_value(metrics.get("Profit Factor")),
                win_rate=format_value(metrics.get("Win Rate (%)")),
                max_dd=format_value(metrics.get("Max Drawdown (%)")),
                trades=format_value(metrics.get("Number of Trades"), digits=0),
                avg_win=format_value(metrics.get("Average Win (%)")),
                avg_loss=format_value(metrics.get("Average Loss (%)")),
                buy_hold=format_value(metrics.get("Buy and Hold Return (%)")),
            )
        )

    lines.append("")
    return "\n".join(lines)


def print_stdout_table(summary_rows: list[dict[str, Any]]) -> None:
    """Print comparison table to stdout."""
    if not summary_rows:
        print("No scenarios were evaluated.")
        return

    display_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        metrics = row.get("metrics", {})
        display_rows.append(
            {
                "strategy": row["strategy"],
                "asset": row["asset"],
                "tf": row["timeframe"],
                "status": row["status"],
                "total_return_%": format_value(metrics.get("Total Return (%)")),
                "sharpe": format_value(metrics.get("Sharpe Ratio")),
                "profit_factor": format_value(metrics.get("Profit Factor")),
                "win_rate_%": format_value(metrics.get("Win Rate (%)")),
                "max_dd_%": format_value(metrics.get("Max Drawdown (%)")),
                "trades": format_value(metrics.get("Number of Trades"), digits=0),
                "buy_hold_%": format_value(metrics.get("Buy and Hold Return (%)")),
            }
        )

    table = pd.DataFrame(display_rows)
    print(table.to_string(index=False))


def run_scenario(
    strategy: StrategySpec,
    asset: str,
    timeframe: str,
    data: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series]:
    """Run one strategy/asset/timeframe scenario end-to-end."""
    scenario: dict[str, Any] = {
        "strategy": strategy.name,
        "asset": asset,
        "timeframe": timeframe,
        "status": "ok",
        "windows_total": 0,
        "windows_success": 0,
        "windows_failed": 0,
        "errors": [],
        "metrics": {},
    }

    if strategy.intraday_only and timeframe != "1h":
        scenario["status"] = "skipped"
        scenario["errors"] = ["Strategy requires intraday data; skipping non-1h timeframe"]
        return scenario, pd.Series(dtype=float)

    windows = make_windows(data)
    if not windows:
        scenario["status"] = "skipped"
        scenario["errors"] = ["No valid walk-forward windows available"]
        return scenario, pd.Series(dtype=float)

    scenario["windows_total"] = len(windows)

    successful_runs: list[WindowRun] = []
    all_trade_samples: list[tuple[pd.Timestamp, float]] = []

    for window in windows:
        try:
            run = strategy.runner(window.train_df, window.test_df)
        except Exception as exc:
            scenario["windows_failed"] += 1
            scenario["errors"].append(
                {
                    "test_start": window.test_start.isoformat(),
                    "test_end": window.test_end.isoformat(),
                    "error": str(exc),
                }
            )
            continue

        scenario["windows_success"] += 1
        successful_runs.append(run)
        all_trade_samples.extend(run.trade_samples)

    if not successful_runs:
        scenario["status"] = "failed"
        scenario["errors"].append("All walk-forward windows failed")
        return scenario, pd.Series(dtype=float)

    equity_curve = concat_window_curves(successful_runs)
    buy_hold_pct = buy_and_hold_return(windows)
    metrics = compute_metrics(equity_curve, all_trade_samples, timeframe, buy_hold_pct)

    scenario["metrics"] = {key: clean_number(value) for key, value in metrics.items()}
    return scenario, equity_curve


def main() -> None:
    """Entrypoint for the unified backtest harness."""
    configure_runtime()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    strategies = [
        StrategySpec(name="FVG", runner=run_fvg_window),
        StrategySpec(name="Williams %R", runner=run_williams_window),
        StrategySpec(name="SMA/EMA Crossover", runner=run_crossover_window),
        StrategySpec(name="Range S/R Breakout", runner=run_breakout_window, intraday_only=True),
        StrategySpec(name="Markov HMM (run_markov_strategy)", runner=run_markov_hmm_window),
        StrategySpec(name="Markov Detailed (run_markov_strategy_detailed)", runner=run_markov_detailed_window),
    ]

    downloaded: dict[tuple[str, str], pd.DataFrame] = {}
    download_errors: dict[str, str] = {}

    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            combo_key = (asset, timeframe)
            try:
                downloaded[combo_key] = download_data(asset, timeframe)
                logging.info("Downloaded %s %s (%d bars)", asset, timeframe, len(downloaded[combo_key]))
            except Exception as exc:
                msg = str(exc)
                download_errors[f"{asset}|{timeframe}"] = msg
                logging.warning("Skipping %s %s: %s", asset, timeframe, msg)

    summary_rows: list[dict[str, Any]] = []
    equity_curves_payload: dict[str, list[dict[str, Any]]] = {}

    for strategy in strategies:
        for asset in ASSETS:
            for timeframe in TIMEFRAMES:
                scenario_key = f"{strategy.name}|{asset}|{timeframe}"
                combo_key = (asset, timeframe)

                if combo_key not in downloaded:
                    row = {
                        "strategy": strategy.name,
                        "asset": asset,
                        "timeframe": timeframe,
                        "status": "skipped",
                        "windows_total": 0,
                        "windows_success": 0,
                        "windows_failed": 0,
                        "errors": [f"Data unavailable: {download_errors.get(f'{asset}|{timeframe}', 'download failed')}"] ,
                        "metrics": {},
                    }
                    summary_rows.append(row)
                    equity_curves_payload[scenario_key] = []
                    continue

                scenario, equity_curve = run_scenario(strategy, asset, timeframe, downloaded[combo_key])
                summary_rows.append(scenario)

                if equity_curve.empty:
                    equity_curves_payload[scenario_key] = []
                else:
                    equity_curves_payload[scenario_key] = [
                        {"timestamp": ts.isoformat(), "equity": clean_number(value, precision=8)}
                        for ts, value in equity_curve.items()
                    ]

    summary_rows.sort(key=lambda row: (row["strategy"], row["asset"], row["timeframe"]))

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "config": {
            "assets": ASSETS,
            "timeframes": TIMEFRAMES,
            "start": START_TS.isoformat(),
            "end": END_TS.isoformat(),
            "walk_forward": {
                "train_months": TRAIN_MONTHS,
                "test_months": TEST_MONTHS,
                "step_months": STEP_MONTHS,
            },
        },
        "download_errors": download_errors,
        "results": summary_rows,
    }

    with (RESULTS_DIR / "backtest_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with (RESULTS_DIR / "equity_curves.json").open("w", encoding="utf-8") as handle:
        json.dump(equity_curves_payload, handle, indent=2)

    markdown = build_markdown(summary_rows)
    (RESULTS_DIR / "RESULTS.md").write_text(markdown, encoding="utf-8")

    print_stdout_table(summary_rows)


if __name__ == "__main__":
    main()
