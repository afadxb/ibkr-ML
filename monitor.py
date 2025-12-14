# monitor.py
import sqlite3
import json
from datetime import datetime, timezone, date, timedelta

import numpy as np
import pandas as pd

from config import TrainConfig
from db_schema import ensure_schema

def load_bars(conn, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, index_col="timestamp", parse_dates=True)
    df.sort_index(inplace=True)
    return df

def expected_bars_per_day(bar_hours: int = 2) -> int:
    # RTH 6.5h => ~3 bars for 2h, but IB bars may align differently.
    # Use a pragmatic expectation for monitoring: 3 bars/day (RTH).
    return 3 if bar_hours == 2 else 2

def data_quality(df: pd.DataFrame, bar_hours: int) -> dict:
    if df.empty:
        return {"missing_bars_pct": 100.0, "outlier_bars_pct": 0.0}

    # last 30 calendar days slice
    end = df.index.max()
    start = end - pd.Timedelta(days=30)
    d = df.loc[df.index >= start].copy()
    if d.empty:
        return {"missing_bars_pct": 100.0, "outlier_bars_pct": 0.0}

    # missing bars estimate by trading day
    d["day"] = d.index.date
    counts = d.groupby("day").size()
    exp = expected_bars_per_day(bar_hours)
    missing = (counts < exp).mean() * 100.0

    # outliers: range z-score
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    z = (rng - rng.mean()) / (rng.std() if rng.std() else 1.0)
    outlier = (np.abs(z) > 4).mean() * 100.0

    return {"missing_bars_pct": float(missing), "outlier_bars_pct": float(outlier)}

def latest_registry(conn, ticker: str) -> dict:
    cur = conn.execute(
        "SELECT model_version, trained_at_utc, holdout_metrics_json, wfo_metrics_json "
        "FROM model_registry WHERE ticker=? ORDER BY id DESC LIMIT 1",
        (ticker,)
    )
    row = cur.fetchone()
    if not row:
        return {}
    mv, trained, holdout_json, wfo_json = row
    hold = json.loads(holdout_json) if holdout_json else {}
    wfo = json.loads(wfo_json) if wfo_json else {}
    return {"model_version": mv, "trained_at_utc": trained, "holdout": hold, "wfo": wfo}

def write_daily(conn, ticker: str, dq: dict, reg: dict, day: str):
    # Use holdout as a proxy for current health until you log live trades/signals
    hold = reg.get("holdout", {})
    signals = hold.get("signals")
    avg_proba = None
    proba_std = None

    precision = hold.get("precision")
    expectancy = hold.get("expectancy_net")
    pf = hold.get("profit_factor_net")
    win_rate = hold.get("win_rate")

    drift_flag = "OK"
    action = "NONE"
    notes = ""

    # simple rule-based flags (can be improved)
    if dq["missing_bars_pct"] > 20:
        drift_flag = "WARN"; notes += "MissingBars>20%; "
    if dq["outlier_bars_pct"] > 5:
        drift_flag = "WARN"; notes += "Outliers>5%; "
    if expectancy is not None and isinstance(expectancy, (int,float)) and expectancy != expectancy:
        expectancy = None

    conn.execute(
        """INSERT INTO monitoring_metrics_daily
           (date,ticker,signals,avg_proba,proba_std,precision,expectancy_net,profit_factor_net,win_rate,
            missing_bars_pct,outlier_bars_pct,drift_flag,action,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (day, ticker, signals, avg_proba, proba_std, precision, expectancy, pf, win_rate,
         dq["missing_bars_pct"], dq["outlier_bars_pct"], drift_flag, action, notes.strip())
    )

def main():
    cfg = TrainConfig()
    ensure_schema(cfg.DATA_DB)

    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        day = date.today().isoformat()
        bar_hours = 2

        for t in cfg.TICKERS:
            try:
                bars = load_bars(conn, t)
                dq = data_quality(bars, bar_hours)
                reg = latest_registry(conn, t)
                write_daily(conn, t, dq, reg, day)
                print(f"{t}: dq={dq} holdout={reg.get('holdout',{})}")
            except Exception as e:
                print(f"{t}: monitor error: {e}")

        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
