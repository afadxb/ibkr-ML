from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from features import compute_supertrend
from utils import utc_now_iso

ST_ATR_GRID = (7, 10, 14, 21)
ST_MULT_GRID = (2.0, 2.5, 3.0, 3.5, 4.0)


def normalize_timeframe_label(bar_size: str) -> str:
    label = bar_size.strip().lower().replace(" ", "")
    label = label.replace("hours", "h").replace("hour", "h")
    return label




@dataclass(frozen=True)
class STRow:
    symbol: str
    timeframe: str
    atr_len: int
    mult: float
    score: float
    median_e: float | None
    median_mdd: float | None
    median_flip_rate: float | None
    fold_count: int
    as_of: str
    run_id: str
    cost_bps: float
    notes: str | None


def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    return float(dd.min()) if not dd.empty else 0.0


def _flip_rate_per_day(df: pd.DataFrame) -> float:
    flips = float(df["st_flip"].sum())
    if df.empty:
        return 0.0
    span_days = max(1.0, (df.index[-1] - df.index[0]).days + 1)
    return flips / span_days


def _score_fold(df_va: pd.DataFrame, cost_bps: float) -> float:
    if len(df_va) < 10:
        return -999.0

    net = df_va.get("st_net_ret")
    if net is None:
        ret = df_va["Close"].pct_change()
        pos = df_va["st_direction"].shift(1).fillna(0)
        gross = ret * pos
        flip_cost = (df_va["st_direction"].shift(1) != df_va["st_direction"]).astype(float)
        net = gross - flip_cost * (cost_bps / 10000.0)

    if net is None or net.dropna().empty:
        return -999.0

    E = float(net.mean())
    pos_sum = float(net[net > 0].sum())
    neg_sum = float(net[net < 0].sum())
    PF = pos_sum / abs(neg_sum) if neg_sum != 0 else 2.5
    WR = float((net > 0).mean()) if len(net.dropna()) else 0.0
    equity = (1 + net.fillna(0)).cumprod()
    MDD = abs(_max_drawdown(equity))
    flip_rate = _flip_rate_per_day(df_va)

    if df_va["st_flip"].sum() < 2:
        return -999.0

    E_clip = np.clip(E, -0.5, 1.0)
    PF_clip = np.clip(PF, 0.8, 2.5)
    WR_clip = np.clip(WR, 0.35, 0.70)
    MDD_clip = np.clip(MDD, 0.5, 8.0)
    FR_clip = np.clip(flip_rate, 0.1, 5.0)

    score = (
        1.4 * E_clip
        + 0.3 * math.log(PF_clip)
        + 0.2 * (WR_clip - 0.5)
        - 0.9 * math.log(1 + MDD_clip)
        - 0.6 * math.log(1 + FR_clip)
    )
    return float(score)


def _wf_folds(idx: pd.DatetimeIndex, train_days: int = 120, val_days: int = 20, step_days: int = 20) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = pd.to_datetime(idx)
    if idx.empty:
        return []
    start = idx.min()
    last = idx.max()
    folds = []
    while True:
        train_end = start + timedelta(days=train_days)
        val_end = train_end + timedelta(days=val_days)
        if val_end > last:
            break
        folds.append((train_end, val_end))
        start = start + timedelta(days=step_days)
    return folds


def select_supertrend_params(
    df: pd.DataFrame,
    cost_bps: float,
    timeframe: str,
    symbol: str,
    run_id: str | None = None,
) -> STRow | None:
    if df.empty:
        return None

    folds = _wf_folds(df.index)
    if not folds:
        return None

    candidates = []
    for atr_len in ST_ATR_GRID:
        for mult in ST_MULT_GRID:
            st_df = compute_supertrend(df, atr_len=atr_len, mult=mult)
            ret = st_df["Close"].pct_change()
            pos = st_df["st_direction"].shift(1).fillna(0)
            gross = ret * pos
            flip_cost = (st_df["st_direction"].shift(1) != st_df["st_direction"]).astype(float)
            st_df["st_net_ret"] = gross - flip_cost * (cost_bps / 10000.0)
            scores = []
            e_vals = []
            mdds = []
            flips = []
            for train_end, val_end in folds:
                va = st_df[(st_df.index >= train_end) & (st_df.index < val_end)]
                if va.empty:
                    continue
                s = _score_fold(va, cost_bps)
                if s <= -900:
                    continue
                scores.append(s)
                e_vals.append(float(va["st_net_ret"].mean()))
                equity = (1 + va["st_net_ret"].fillna(0)).cumprod()
                mdds.append(abs(_max_drawdown(equity)))
                flips.append(_flip_rate_per_day(va))

            if not scores:
                continue

            med_score = float(np.nanmedian(scores))
            stability_penalty = 0.25 * float(np.nanstd(scores))
            final_score = med_score - stability_penalty

            candidates.append({
                "atr_len": atr_len,
                "mult": mult,
                "score": final_score,
                "median_e": float(np.nanmedian(e_vals)) if e_vals else None,
                "median_mdd": float(np.nanmedian(mdds)) if mdds else None,
                "median_flip_rate": float(np.nanmedian(flips)) if flips else None,
                "fold_count": len(scores),
                "raw_scores": scores,
            })

    if not candidates:
        return None

    def _tie_key(c: dict) -> tuple:
        return (
            -c["score"],
            c.get("median_mdd") if c.get("median_mdd") is not None else float("inf"),
            c.get("median_flip_rate") if c.get("median_flip_rate") is not None else float("inf"),
            -(c.get("median_e") if c.get("median_e") is not None else -float("inf")),
        )

    best = sorted(candidates, key=_tie_key)[0]
    as_of = utc_now_iso()
    return STRow(
        symbol=symbol,
        timeframe=timeframe,
        atr_len=best["atr_len"],
        mult=best["mult"],
        score=best["score"],
        median_e=best.get("median_e"),
        median_mdd=best.get("median_mdd"),
        median_flip_rate=best.get("median_flip_rate"),
        fold_count=best["fold_count"],
        as_of=as_of,
        run_id=run_id or f"stgrid_{symbol}_{timeframe}_{as_of}",
        cost_bps=cost_bps,
        notes=None,
    )


def save_st_recommendations(db_path: str, rows: Iterable[STRow]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for r in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO st_param_recommendations
                (symbol,timeframe,atr_len,mult,score,median_e,median_mdd,median_flip_rate,fold_count,as_of,run_id,cost_bps,notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    r.symbol,
                    r.timeframe,
                    r.atr_len,
                    r.mult,
                    r.score,
                    r.median_e,
                    r.median_mdd,
                    r.median_flip_rate,
                    r.fold_count,
                    r.as_of,
                    r.run_id,
                    r.cost_bps,
                    r.notes,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_latest_st_params(db_path: str, symbol: str, timeframe: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT * FROM st_param_recommendations
            WHERE symbol=? AND timeframe=?
            ORDER BY as_of DESC
            LIMIT 1
            """,
            (symbol, timeframe),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def resolve_st_params(db_path: str, symbol: str, timeframe: str, default_period: int, default_mult: float) -> dict:
    rec = get_latest_st_params(db_path, symbol, timeframe)
    if rec:
        return {"atr_len": int(rec["atr_len"]), "mult": float(rec["mult"]), "meta": rec}
    return {"atr_len": default_period, "mult": default_mult, "meta": None}
