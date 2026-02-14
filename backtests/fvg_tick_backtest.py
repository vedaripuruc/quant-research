#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PIP_MULTIPLIER = 10_000.0
MIN_GAP = 0.0001  # 1 pip

DATA_DIR = Path("tickdata/EURUSD")
RESULTS_DIR = Path("results")

DATA_START = datetime(2023, 7, 6, 0, 0, tzinfo=timezone.utc)
DATA_END = datetime(2026, 2, 10, 23, 0, tzinfo=timezone.utc)

WF_PERIOD_START = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
WF_PERIOD_END = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
TRAIN_MONTHS = 6
OOS_MONTHS = 2


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int


@dataclass
class PendingOrder:
    order_id: int
    fvg_time_ms: int
    fvg_type: str  # bullish | bearish
    midpoint: float
    gap_size: float
    zone_start: float
    zone_end: float


@dataclass
class ActiveTrade:
    order_id: int
    fvg_time_ms: int
    fvg_type: str  # bullish | bearish
    midpoint: float
    gap_size: float
    entry_time_ms: int
    entry_price: float
    sl: float
    tp: float


@dataclass
class ClosedTrade:
    fvg_time_ms: int
    fvg_type: str
    midpoint: float
    gap_size: float
    entry_time_ms: int
    entry_price: float
    exit_time_ms: int
    exit_price: float
    sl: float
    tp: float
    result: str
    pips: float
    duration_seconds: float


def dt_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def ms_to_iso(value_ms: int) -> str:
    return datetime.fromtimestamp(value_ms / 1000, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def add_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + (value.month - 1) + months
    year = month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month, day=1)


def iter_hours(start: datetime, end: datetime):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(hours=1)


def calculate_max_drawdown_pips(trades: list[ClosedTrade]) -> float:
    if not trades:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_time_ms):
        equity += trade.pips
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def calculate_sharpe_annualized(trades: list[ClosedTrade]) -> float:
    if len(trades) < 2:
        return 0.0
    returns = [trade.pips for trade in trades]
    std_dev = statistics.stdev(returns)
    if std_dev == 0:
        return 0.0

    mean_return = statistics.fmean(returns)
    first_entry = min(trade.entry_time_ms for trade in trades)
    last_exit = max(trade.exit_time_ms for trade in trades)
    elapsed_years = (last_exit - first_entry) / (1000 * 60 * 60 * 24 * 365.25)
    if elapsed_years <= 0:
        return 0.0

    trades_per_year = len(trades) / elapsed_years
    return (mean_return / std_dev) * math.sqrt(trades_per_year)


def calculate_metrics(trades: list[ClosedTrade]) -> dict[str, float | int | None]:
    total = len(trades)
    wins = [trade for trade in trades if trade.pips > 0]
    losses = [trade for trade in trades if trade.pips < 0]

    gross_wins = sum(trade.pips for trade in wins)
    gross_losses_abs = abs(sum(trade.pips for trade in losses))
    profit_factor = None if gross_losses_abs == 0 else gross_wins / gross_losses_abs

    return {
        "total_trades": total,
        "win_rate_pct": (len(wins) / total * 100.0) if total else 0.0,
        "profit_factor": profit_factor,
        "total_return_pips": sum(trade.pips for trade in trades),
        "avg_win_pips": (gross_wins / len(wins)) if wins else 0.0,
        "avg_loss_pips": (sum(trade.pips for trade in losses) / len(losses)) if losses else 0.0,
        "max_drawdown_pips": calculate_max_drawdown_pips(trades),
        "sharpe_annualized": calculate_sharpe_annualized(trades),
    }


def detect_fvg_and_create_order(
    bars: list[Bar], pending_orders: list[PendingOrder], next_order_id: int
) -> int:
    if len(bars) < 3:
        return next_order_id

    left = bars[-3]
    right = bars[-1]

    # Bullish FVG: current low above low-2 high
    if right.low > left.high:
        gap_size = right.low - left.high
        if gap_size >= MIN_GAP:
            pending_orders.append(
                PendingOrder(
                    order_id=next_order_id,
                    fvg_time_ms=dt_to_ms(right.timestamp),
                    fvg_type="bullish",
                    midpoint=(left.high + right.low) / 2.0,
                    gap_size=gap_size,
                    zone_start=left.high,
                    zone_end=right.low,
                )
            )
            return next_order_id + 1

    # Bearish FVG: low-2 low above current high
    if left.low > right.high:
        gap_size = left.low - right.high
        if gap_size >= MIN_GAP:
            pending_orders.append(
                PendingOrder(
                    order_id=next_order_id,
                    fvg_time_ms=dt_to_ms(right.timestamp),
                    fvg_type="bearish",
                    midpoint=(right.high + left.low) / 2.0,
                    gap_size=gap_size,
                    zone_start=left.low,
                    zone_end=right.high,
                )
            )
            return next_order_id + 1

    return next_order_id


def update_orders_and_trades_on_tick(
    ts_ms: int,
    price: float,
    pending_orders: list[PendingOrder],
    active_trades: list[ActiveTrade],
    closed_trades: list[ClosedTrade],
) -> tuple[int, int]:
    fills = 0
    cancellations = 0

    if pending_orders:
        still_pending: list[PendingOrder] = []
        new_active: list[ActiveTrade] = []

        for order in pending_orders:
            filled = False
            cancelled = False

            if order.fvg_type == "bullish":
                if price <= order.midpoint:
                    filled = True
                    fills += 1
                    new_active.append(
                        ActiveTrade(
                            order_id=order.order_id,
                            fvg_time_ms=order.fvg_time_ms,
                            fvg_type=order.fvg_type,
                            midpoint=order.midpoint,
                            gap_size=order.gap_size,
                            entry_time_ms=ts_ms,
                            entry_price=order.midpoint,
                            sl=order.midpoint - 1.5 * order.gap_size,
                            tp=order.midpoint + 3.0 * order.gap_size,
                        )
                    )
                elif price < order.zone_start:
                    cancelled = True
                    cancellations += 1
            else:
                if price >= order.midpoint:
                    filled = True
                    fills += 1
                    new_active.append(
                        ActiveTrade(
                            order_id=order.order_id,
                            fvg_time_ms=order.fvg_time_ms,
                            fvg_type=order.fvg_type,
                            midpoint=order.midpoint,
                            gap_size=order.gap_size,
                            entry_time_ms=ts_ms,
                            entry_price=order.midpoint,
                            sl=order.midpoint + 1.5 * order.gap_size,
                            tp=order.midpoint - 3.0 * order.gap_size,
                        )
                    )
                elif price > order.zone_start:
                    cancelled = True
                    cancellations += 1

            if not filled and not cancelled:
                still_pending.append(order)

        pending_orders[:] = still_pending
        active_trades.extend(new_active)

    if active_trades:
        still_active: list[ActiveTrade] = []
        for trade in active_trades:
            exit_price: float | None = None

            if trade.fvg_type == "bullish":
                if price <= trade.sl:
                    exit_price = trade.sl
                elif price >= trade.tp:
                    exit_price = trade.tp
            else:
                if price >= trade.sl:
                    exit_price = trade.sl
                elif price <= trade.tp:
                    exit_price = trade.tp

            if exit_price is None:
                still_active.append(trade)
                continue

            if trade.fvg_type == "bullish":
                pips = (exit_price - trade.entry_price) * PIP_MULTIPLIER
            else:
                pips = (trade.entry_price - exit_price) * PIP_MULTIPLIER

            closed_trades.append(
                ClosedTrade(
                    fvg_time_ms=trade.fvg_time_ms,
                    fvg_type=trade.fvg_type,
                    midpoint=trade.midpoint,
                    gap_size=trade.gap_size,
                    entry_time_ms=trade.entry_time_ms,
                    entry_price=trade.entry_price,
                    exit_time_ms=ts_ms,
                    exit_price=exit_price,
                    sl=trade.sl,
                    tp=trade.tp,
                    result="Win" if pips > 0 else "Loss",
                    pips=pips,
                    duration_seconds=max(0.0, (ts_ms - trade.entry_time_ms) / 1000.0),
                )
            )

        active_trades[:] = still_active

    return fills, cancellations


def process_hour_file(
    file_path: Path,
    hour_start: datetime,
    pending_orders: list[PendingOrder],
    active_trades: list[ActiveTrade],
    closed_trades: list[ClosedTrade],
) -> tuple[Bar | None, int, int]:
    open_price = 0.0
    high_price = 0.0
    low_price = 0.0
    close_price = 0.0
    volume_sum = 0.0
    tick_count = 0
    file_fills = 0
    file_cancellations = 0

    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 3:
                continue

            try:
                ts_ms = int(parts[0])
                price = float(parts[1])
                volume = float(parts[2])
            except ValueError:
                continue

            fills, cancellations = update_orders_and_trades_on_tick(
                ts_ms=ts_ms,
                price=price,
                pending_orders=pending_orders,
                active_trades=active_trades,
                closed_trades=closed_trades,
            )
            file_fills += fills
            file_cancellations += cancellations

            if tick_count == 0:
                open_price = high_price = low_price = close_price = price
            else:
                if price > high_price:
                    high_price = price
                if price < low_price:
                    low_price = price
                close_price = price

            volume_sum += volume
            tick_count += 1

    if tick_count == 0:
        return None, file_fills, file_cancellations

    return (
        Bar(
            timestamp=hour_start,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume_sum,
            tick_count=tick_count,
        ),
        file_fills,
        file_cancellations,
    )


def build_walk_forward_windows() -> list[dict[str, int | str]]:
    windows: list[dict[str, int | str]] = []
    oos_start = add_months(WF_PERIOD_START, TRAIN_MONTHS)
    window_id = 1

    while oos_start < WF_PERIOD_END:
        oos_end = add_months(oos_start, OOS_MONTHS)
        if oos_end > WF_PERIOD_END:
            oos_end = WF_PERIOD_END

        train_start = add_months(oos_start, -TRAIN_MONTHS)
        windows.append(
            {
                "window_id": window_id,
                "train_start": train_start.isoformat().replace("+00:00", "Z"),
                "train_end": oos_start.isoformat().replace("+00:00", "Z"),
                "oos_start": oos_start.isoformat().replace("+00:00", "Z"),
                "oos_end": oos_end.isoformat().replace("+00:00", "Z"),
                "oos_start_ms": dt_to_ms(oos_start),
                "oos_end_ms": dt_to_ms(oos_end),
            }
        )

        oos_start = add_months(oos_start, OOS_MONTHS)
        window_id += 1

    return windows


def save_trades_csv(trades: list[ClosedTrade], destination: Path) -> None:
    fieldnames = [
        "fvg_time",
        "fvg_type",
        "midpoint",
        "gap_size",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "sl",
        "tp",
        "result",
        "pips",
        "duration_seconds",
    ]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in sorted(trades, key=lambda item: item.entry_time_ms):
            writer.writerow(
                {
                    "fvg_time": ms_to_iso(trade.fvg_time_ms),
                    "fvg_type": trade.fvg_type,
                    "midpoint": f"{trade.midpoint:.5f}",
                    "gap_size": f"{trade.gap_size:.5f}",
                    "entry_time": ms_to_iso(trade.entry_time_ms),
                    "entry_price": f"{trade.entry_price:.5f}",
                    "exit_time": ms_to_iso(trade.exit_time_ms),
                    "exit_price": f"{trade.exit_price:.5f}",
                    "sl": f"{trade.sl:.5f}",
                    "tp": f"{trade.tp:.5f}",
                    "result": trade.result,
                    "pips": f"{trade.pips:.2f}",
                    "duration_seconds": f"{trade.duration_seconds:.2f}",
                }
            )


def print_metrics(title: str, metrics: dict[str, float | int | None]) -> None:
    profit_factor = metrics["profit_factor"]
    pf_display = "inf" if profit_factor is None else f"{profit_factor:.4f}"
    print(f"\n{title}")
    print(f"  Total trades: {metrics['total_trades']}")
    print(f"  Win Rate %: {metrics['win_rate_pct']:.2f}")
    print(f"  Profit Factor: {pf_display}")
    print(f"  Total Return (pips): {metrics['total_return_pips']:.2f}")
    print(f"  Avg Win/Loss (pips): {metrics['avg_win_pips']:.2f} / {metrics['avg_loss_pips']:.2f}")
    print(f"  Max Drawdown (pips): {metrics['max_drawdown_pips']:.2f}")
    print(f"  Sharpe ratio (annualized): {metrics['sharpe_annualized']:.4f}")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Tick data directory not found: {DATA_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bars: list[Bar] = []
    pending_orders: list[PendingOrder] = []
    active_trades: list[ActiveTrade] = []
    closed_trades: list[ClosedTrade] = []

    files_processed = 0
    files_missing = 0
    files_empty = 0
    total_fills = 0
    total_cancellations = 0
    next_order_id = 1

    total_hours = int((DATA_END - DATA_START).total_seconds() // 3600) + 1
    print(f"Scanning hourly files from {DATA_START.date()} to {DATA_END.date()} ({total_hours} hours)")

    for hour_start in iter_hours(DATA_START, DATA_END):
        file_name = f"EURUSD_BID_{hour_start:%Y-%m-%d_%H}.log.gz"
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            files_missing += 1
            continue

        bar, fills, cancellations = process_hour_file(
            file_path=file_path,
            hour_start=hour_start,
            pending_orders=pending_orders,
            active_trades=active_trades,
            closed_trades=closed_trades,
        )
        files_processed += 1
        total_fills += fills
        total_cancellations += cancellations

        if files_processed % 100 == 0:
            print(f"Processed {files_processed} files... ({hour_start:%Y-%m-%d %H:00} UTC)")

        if bar is None:
            files_empty += 1
            continue

        bars.append(bar)
        next_order_id = detect_fvg_and_create_order(bars, pending_orders, next_order_id)

    windows = build_walk_forward_windows()
    aggregate_start_ms = dt_to_ms(add_months(WF_PERIOD_START, TRAIN_MONTHS))
    aggregate_end_ms = dt_to_ms(WF_PERIOD_END)

    aggregate_oos_trades = [
        trade
        for trade in closed_trades
        if aggregate_start_ms <= trade.entry_time_ms < aggregate_end_ms
    ]
    aggregate_metrics = calculate_metrics(aggregate_oos_trades)

    window_reports: list[dict[str, object]] = []
    for window in windows:
        window_trades = [
            trade
            for trade in closed_trades
            if window["oos_start_ms"] <= trade.entry_time_ms < window["oos_end_ms"]
        ]
        metrics = calculate_metrics(window_trades)
        report = {
            "window_id": window["window_id"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "oos_start": window["oos_start"],
            "oos_end": window["oos_end"],
            "metrics": metrics,
        }
        window_reports.append(report)

    trades_csv_path = RESULTS_DIR / "fvg_tick_trades.csv"
    summary_json_path = RESULTS_DIR / "fvg_tick_summary.json"
    save_trades_csv(closed_trades, trades_csv_path)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "data_start": DATA_START.isoformat().replace("+00:00", "Z"),
            "data_end": DATA_END.isoformat().replace("+00:00", "Z"),
            "min_gap": MIN_GAP,
            "pip_multiplier": PIP_MULTIPLIER,
            "walk_forward_period_start": WF_PERIOD_START.isoformat().replace("+00:00", "Z"),
            "walk_forward_period_end": WF_PERIOD_END.isoformat().replace("+00:00", "Z"),
            "train_months": TRAIN_MONTHS,
            "oos_months": OOS_MONTHS,
        },
        "processing": {
            "files_processed": files_processed,
            "files_missing": files_missing,
            "files_empty": files_empty,
            "bars_built": len(bars),
            "fvg_zones_detected": next_order_id - 1,
            "fills": total_fills,
            "cancellations": total_cancellations,
            "pending_orders_open_at_end": len(pending_orders),
            "active_trades_open_at_end": len(active_trades),
            "closed_trades_total": len(closed_trades),
        },
        "aggregate_oos": {
            "period_start": ms_to_iso(aggregate_start_ms),
            "period_end": ms_to_iso(aggregate_end_ms),
            "metrics": aggregate_metrics,
        },
        "per_window": window_reports,
    }

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Tick Processing Summary ===")
    print(f"Existing files processed: {files_processed}")
    print(f"Missing files skipped: {files_missing}")
    print(f"Empty files skipped: {files_empty}")
    print(f"Bars built: {len(bars)}")
    print(f"FVG zones detected: {next_order_id - 1}")
    print(f"Closed trades (all data): {len(closed_trades)}")
    print(f"Open pending orders at end: {len(pending_orders)}")
    print(f"Open active trades at end: {len(active_trades)}")

    print_metrics("=== Aggregate OOS Metrics ===", aggregate_metrics)
    print("\n=== Per-Window OOS Metrics ===")
    for report in window_reports:
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"Window {report['window_id']} | "
            f"Train {report['train_start']} -> {report['train_end']} | "
            f"OOS {report['oos_start']} -> {report['oos_end']}"
        )
        print(
            "  Trades={total_trades}, WinRate={win_rate:.2f}%, PF={pf}, "
            "Return={ret:.2f} pips, MaxDD={dd:.2f} pips, Sharpe={sharpe:.4f}".format(
                total_trades=metrics["total_trades"],
                win_rate=metrics["win_rate_pct"],
                pf="inf" if metrics["profit_factor"] is None else f"{metrics['profit_factor']:.4f}",
                ret=metrics["total_return_pips"],
                dd=metrics["max_drawdown_pips"],
                sharpe=metrics["sharpe_annualized"],
            )
        )

    print("\nSaved outputs:")
    print(f"  Trades CSV: {trades_csv_path}")
    print(f"  Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
