from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import pandas as pd


def _align_to_bar(target: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Snap a timestamp to the most recent bar close in ``idx``.

    Fold boundaries should always align to actual bar timestamps (RTH only),
    not arbitrary wall-clock days. This helper returns the latest bar close
    at or before ``target`` or ``None`` if ``target`` precedes the history.
    """
    pos = idx.searchsorted(target, side="right")
    if pos == 0:
        return None
    return idx[pos - 1]

@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

def timestamp_folds(
    index: pd.DatetimeIndex,
    n_folds: int,
    val_days: int,
    gap_days: int,
    min_train_bars: int,
    mode: str = "expanding",
    rolling_train_months: int = 18
) -> list[Fold]:
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) < (min_train_bars + 100):
        return []

    end = idx.max()
    start = idx.min()

    val = timedelta(days=val_days)
    gap = timedelta(days=gap_days)

    # Use historical RTH cadence to require sufficiently dense validation windows
    bars_per_day = idx.normalize().value_counts()
    typical_bars_per_day = int(bars_per_day.median()) if not bars_per_day.empty else 0
    if typical_bars_per_day == 0:
        return []
    min_val_bars = max(50, int(typical_bars_per_day * val_days * 0.6))

    # choose fold validation ends evenly spaced in the eligible range
    # Eligible latest val_end leaves no future; we let caller keep holdout separate.
    # Here we build folds within the provided index range.
    # Start building from the end backwards.
    folds = []
    # pick n_folds validation windows ending at evenly spaced points
    # simple approach: compute candidate val_end timestamps by percentile positions
    positions = [int(len(idx) * (0.5 + 0.5*(k+1)/n_folds)) for k in range(n_folds)]
    positions = [min(len(idx)-1, max(0, p)) for p in positions]

    for p in positions:
        val_end = idx[p]
        val_start = _align_to_bar(val_end - val, idx)
        gap_end = val_start
        gap_start = _align_to_bar(gap_end - gap, idx) if gap_end is not None else None
        train_end = gap_start

        if val_start is None or gap_start is None:
            continue

        if mode == "rolling":
            train_start = _align_to_bar(train_end - pd.DateOffset(months=rolling_train_months), idx)
        else:
            train_start = start

        if train_start is None:
            continue

        # enforce ordering
        if train_end <= train_start:
            continue

        # enforce bar counts
        train_mask = (idx >= train_start) & (idx < train_end)
        val_mask = (idx >= val_start) & (idx < val_end)
        if train_mask.sum() < min_train_bars:
            continue
        if val_mask.sum() < min_val_bars:
            continue

        folds.append(Fold(
            train_start=train_start, train_end=train_end,
            gap_start=gap_start, gap_end=gap_end,
            val_start=val_start, val_end=val_end
        ))

    # sort by val_start ascending and de-dup overlapping by time (optional)
    folds = sorted(folds, key=lambda f: f.val_start)
    return folds
