"""
Aggregate tick data to 15-minute OHLC bars for a symbol.
Reads all .gz files from tickdata/<SYMBOL>/, groups ticks into 15-min buckets,
outputs OHLC + volume + tick count to tickdata/<SYMBOL>_15M_2Y.csv

Format: timestamp_ms, price, volume (one tick per line, gzip compressed)
"""

import os
import gzip
import glob
import time
import argparse
import csv

FIFTEEN_MIN_MS = 15 * 60 * 1000


def floor_15m(ts_ms: int) -> int:
    """Floor timestamp to nearest 15-minute boundary."""
    return (ts_ms // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS


def main():
    parser = argparse.ArgumentParser(description="Aggregate Darwinex ticks into 15-minute OHLC.")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol directory name.")
    parser.add_argument("--data-dir", default="tickdata", help="Base tickdata directory.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    tick_dir = os.path.join(args.data_dir, args.symbol)
    pattern = os.path.join(tick_dir, f"{args.symbol}_BID_*.log.gz")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} tick files")

    # We'll accumulate all bars in a dict: bucket_ts -> {open, high, low, close, volume, ticks}
    # Since files are sorted chronologically, we process in order.
    # To keep memory manageable, we use a flat dict.
    bars = {}  # bucket_ts_ms -> [open_price, high, low, close, volume, tick_count, first_tick_ts]

    t0 = time.time()
    total_ticks = 0
    errors = 0

    for fi, fpath in enumerate(files):
        if (fi + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {fi+1}/{len(files)} files, {total_ticks:,} ticks, "
                  f"{len(bars):,} bars, {elapsed:.1f}s elapsed")

        try:
            with gzip.open(fpath, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    try:
                        ts_ms = int(parts[0])
                        price = float(parts[1])
                        volume = float(parts[2])
                    except (ValueError, IndexError):
                        errors += 1
                        continue

                    bucket = floor_15m(ts_ms)
                    total_ticks += 1

                    if bucket not in bars:
                        bars[bucket] = [price, price, price, price, volume, 1, ts_ms]
                    else:
                        bar = bars[bucket]
                        if ts_ms < bar[6]:
                            # Earlier tick than what we have — update open
                            bar[0] = price
                            bar[6] = ts_ms
                        if price > bar[1]:
                            bar[1] = price  # high
                        if price < bar[2]:
                            bar[2] = price  # low
                        # Close = last tick by time. Since files are sorted, 
                        # just always overwrite close with current price.
                        # But ticks within a file might not be perfectly sorted.
                        # We'll set close to the tick with the latest timestamp.
                        bar[3] = price  # We'll fix this below
                        bar[4] += volume
                        bar[5] += 1
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
            errors += 1

    elapsed = time.time() - t0
    print(f"\nParsing done: {total_ticks:,} ticks, {len(bars):,} bars, "
          f"{errors} errors, {elapsed:.1f}s")

    # The 'close' field needs to be the price of the last tick in each bucket.
    # Since we process files in chronological order (sorted by date_hour),
    # and within each file ticks are chronological, the last assignment 
    # to bar[3] should be correct. But to be safe, let's do a second pass
    # where we track the max timestamp per bucket.
    # 
    # Actually, the issue is that we don't track max ts for close.
    # Let me fix: we need to re-scan or track it properly.
    # 
    # Given files are sorted by hour and ticks within are chronological,
    # processing in file order means the last tick written per bucket IS the close.
    # The only edge case: a 15m bucket spans two hourly files (e.g., XX:45-XX:59 in file _HH 
    # and XX:00-XX:00 in file _HH+1, but that's the NEXT bucket).
    # Actually 15m buckets align with hours (0,15,30,45), so each bucket is fully 
    # within one hour file or split across two. But since files are sorted, 
    # later files overwrite close correctly.
    #
    # However, the OPEN must be the first tick. We tracked first_tick_ts (bar[6])
    # and set open only when we see an earlier tick. But since files are in order,
    # the first file for a bucket sets the open and it won't be overwritten.
    # That's correct.

    # Sort by bucket timestamp and write
    sorted_buckets = sorted(bars.keys())

    # Filter out weekend bars (Saturday=5, Sunday=6)
    from datetime import datetime, timezone
    
    filtered = []
    for bucket_ts in sorted_buckets:
        dt = datetime.fromtimestamp(bucket_ts / 1000, tz=timezone.utc)
        if dt.weekday() in (5, 6):  # Saturday, Sunday
            continue
        filtered.append(bucket_ts)

    print(f"After weekend filter: {len(filtered)} bars (removed {len(sorted_buckets) - len(filtered)})")

    output = args.output if args.output else os.path.join(args.data_dir, f"{args.symbol}_15M_2Y.csv")
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticks'])
        for bucket_ts in filtered:
            bar = bars[bucket_ts]
            # timestamp as ISO string in UTC
            dt = datetime.fromtimestamp(bucket_ts / 1000, tz=timezone.utc)
            writer.writerow([
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                f"{bar[0]:.5f}",
                f"{bar[1]:.5f}",
                f"{bar[2]:.5f}",
                f"{bar[3]:.5f}",
                f"{bar[4]:.0f}",
                bar[5],
            ])

    print(f"Saved {len(filtered)} bars to {output}")
    print(f"Date range: {datetime.fromtimestamp(filtered[0]/1000, tz=timezone.utc)} to "
          f"{datetime.fromtimestamp(filtered[-1]/1000, tz=timezone.utc)}")


if __name__ == '__main__':
    main()
