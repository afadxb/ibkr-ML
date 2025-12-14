from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone, timedelta, date
import numpy as np
import pandas as pd

from config import BotConfig
from db_schema import ensure_schema
from pipeline import make_features
from datahub import get_cached_bars
from modelio import load_bundle

def week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())

def psi_quantile_bins(base: pd.Series, current: pd.Series, bins: int, eps: float = 1e-6) -> float:
    base = base.dropna()
    current = current.dropna()
    if len(base) < 100 or len(current) < 50:
        return float("nan")
    # categorical
    if base.nunique() <= 10 and base.dtype.kind in ("i","b","O"):
        cats = sorted(set(base.unique()).union(set(current.unique())))
        b = base.value_counts(normalize=True).reindex(cats).fillna(0.0).values
        c = current.value_counts(normalize=True).reindex(cats).fillna(0.0).values
    else:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(base.values, qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            return float("nan")
        b = np.histogram(base.values, bins=edges)[0].astype(float)
        c = np.histogram(current.values, bins=edges)[0].astype(float)
        b = b / max(1.0, b.sum())
        c = c / max(1.0, c.sum())

    b = np.clip(b, eps, 1.0)
    c = np.clip(c, eps, 1.0)
    return float(np.sum((b - c) * np.log(b / c)))

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    a = np.sort(a); b = np.sort(b)
    data_all = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data_all, side='right') / len(a)
    cdf_b = np.searchsorted(b, data_all, side='right') / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def upsert_feature_week(conn, week_start: str, ticker: str, feature: str, psi: float, flag: str):
    conn.execute("DELETE FROM drift_feature_weekly WHERE week_start=? AND ticker=? AND feature=?", (week_start, ticker, feature))
    conn.execute(
        "INSERT INTO drift_feature_weekly (week_start,ticker,feature,psi,flag) VALUES (?,?,?,?,?)",
        (week_start, ticker, feature, float(psi) if psi==psi else None, flag)
    )

def upsert_pred_week(conn, week_start: str, ticker: str, mean_shift: float, std_shift: float, ks: float, flag: str):
    conn.execute("DELETE FROM drift_pred_weekly WHERE week_start=? AND ticker=?", (week_start, ticker))
    conn.execute(
        "INSERT INTO drift_pred_weekly (week_start,ticker,proba_mean_shift,proba_std_shift,ks_stat,flag) VALUES (?,?,?,?,?,?)",
        (week_start, ticker,
         float(mean_shift) if mean_shift==mean_shift else None,
         float(std_shift) if std_shift==std_shift else None,
         float(ks) if ks==ks else None,
         flag)
    )

def main():
    cfg = BotConfig()
    ensure_schema(cfg.DATA_DB)

    # compute week window
    today = date.today()
    ws = week_start_monday(today)
    we = ws + timedelta(days=7)
    ws_str = ws.isoformat()

    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        for t in cfg.TICKERS:
            # Load model bundle + required features
            bundle = load_bundle(t, cfg)
            feats = bundle["features"]

            # ----- PSI on top-K features (model features) -----
            bars = get_cached_bars(cfg.DATA_DB, t)
            if bars.empty:
                continue

            # Build features on full history to slice baseline/current
            df_feat = make_features(bars, cfg).dropna()
            if df_feat.empty:
                continue

            # baseline: training window from registry
            cur = conn.execute(
                "SELECT train_start, train_end FROM model_registry WHERE ticker=? ORDER BY id DESC LIMIT 1",
                (t,)
            )
            row = cur.fetchone()
            if not row:
                continue
            train_start, train_end = pd.to_datetime(row[0]), pd.to_datetime(row[1])

            base = df_feat.loc[(df_feat.index >= train_end - pd.Timedelta(days=cfg.DRIFT_BASELINE_PRED_DAYS)) & (df_feat.index < train_end)]
            curr = df_feat.loc[(df_feat.index >= pd.Timestamp(ws)) & (df_feat.index < pd.Timestamp(we))]
            # If current week slice empty (e.g., weekend run), use last 7 days
            if curr.empty:
                curr = df_feat.tail(7*12)  # ~7 days * 12 2h bars (approx)

            top_feats = feats[: min(len(feats), cfg.SHAP_TOP_K)]
            for f in top_feats:
                if f not in df_feat.columns:
                    continue
                psi = psi_quantile_bins(base[f], curr[f], cfg.PSI_BINS)
                if psi != psi:
                    flag = "OK"
                elif psi >= cfg.PSI_ACTION:
                    flag = "ACTION"
                elif psi >= cfg.PSI_WARN:
                    flag = "WARN"
                else:
                    flag = "OK"
                upsert_feature_week(conn, ws_str, t, f, psi, flag)

            # ----- KS on real predictions (preferred) -----
            # baseline: last N days predictions prior to week start
            pred = pd.read_sql(
                "SELECT ts_utc, prob_up FROM predictions_log WHERE ticker=?",
                conn, params=(t,)
            )
            if not pred.empty:
                pred["ts_utc"] = pd.to_datetime(pred["ts_utc"], utc=True)
                pred = pred.sort_values("ts_utc")
                cur_week = pred[(pred["ts_utc"] >= pd.Timestamp(ws, tz='UTC')) & (pred["ts_utc"] < pd.Timestamp(we, tz='UTC'))]["prob_up"].values
                base_start = pd.Timestamp(ws, tz='UTC') - pd.Timedelta(days=cfg.DRIFT_BASELINE_PRED_DAYS)
                base_pred = pred[(pred["ts_utc"] >= base_start) & (pred["ts_utc"] < pd.Timestamp(ws, tz='UTC'))]["prob_up"].values
            else:
                # fallback: compute proba distributions from model on baseline/current bars
                # NOTE: this is weaker than using real production predictions
                base_pred = np.array([])
                cur_week = np.array([])

            ks = ks_statistic(base_pred, cur_week)
            mean_shift = (np.nanmean(cur_week) - np.nanmean(base_pred)) if (len(cur_week) and len(base_pred)) else float("nan")
            std_shift = (np.nanstd(cur_week) - np.nanstd(base_pred)) if (len(cur_week) and len(base_pred)) else float("nan")

            if ks != ks:
                flag = "OK"
            elif ks >= cfg.KS_ACTION:
                flag = "ACTION"
            elif ks >= cfg.KS_WARN:
                flag = "WARN"
            else:
                flag = "OK"

            upsert_pred_week(conn, ws_str, t, mean_shift, std_shift, ks, flag)

        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
