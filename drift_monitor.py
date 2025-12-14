"""Weekly PSI/KS drift monitoring."""
import json
import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

from config import TrainConfig
from db_schema import ensure_schema
from features import add_features


PSI_THRESHOLDS = {"OK": 0.10, "WARN": 0.20}
KS_THRESHOLDS = {"OK": 0.10, "WARN": 0.20}
EPSILON = 1e-6


FLAG_SEVERITY = {"OK": 0, "WARN": 1, "ACTION": 2}


def completed_week_bounds(today: date | None = None) -> tuple[date, date]:
    """Return (week_start, week_end) for the most recently completed Mon-Sun week."""
    today = today or date.today()
    last_sunday = today - timedelta(days=((today.weekday() + 1) % 7))
    week_end = last_sunday
    week_start = week_end - timedelta(days=6)
    return week_start, week_end


def load_bars(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, index_col="timestamp", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def latest_registry(conn: sqlite3.Connection, ticker: str) -> dict:
    cur = conn.execute(
        """
        SELECT model_version, train_start, train_end, features_checksum
        FROM model_registry
        WHERE ticker=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (ticker,),
    )
    row = cur.fetchone()
    if not row:
        return {}
    mv, tr_start, tr_end, chk = row
    return {"model_version": mv, "train_start": tr_start, "train_end": tr_end, "features_checksum": chk}


def load_feature_list(cfg: TrainConfig, ticker: str, top_k: int | None = None) -> list[str]:
    path = os.path.join(cfg.MODEL_DIR, f"{ticker}_features.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f.readlines() if line.strip()]
    if top_k:
        feats = feats[:top_k]
    return feats


def build_feature_frame(raw: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    df = add_features(raw, cfg.REGIME_ROLL)
    return df


def _categorical_bins(series: pd.Series) -> list:
    cats = sorted(series.dropna().unique().tolist())
    return cats


def _quantile_bins(series: pd.Series, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(series.dropna().quantile(qs).values)
    if len(edges) < 2:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_psi(baseline: pd.Series, current: pd.Series) -> float | None:
    if baseline.empty or current.empty:
        return None

    # decide binning strategy
    is_categorical = pd.api.types.is_integer_dtype(baseline.dropna()) or baseline.dropna().nunique() <= 12
    if is_categorical:
        bins = _categorical_bins(baseline)
        if not bins:
            return None
        base_counts = baseline.value_counts(normalize=True).reindex(bins, fill_value=0.0)
        curr_counts = current.value_counts(normalize=True).reindex(bins, fill_value=0.0)
    else:
        bins = _quantile_bins(baseline, 10)
        if bins.size == 0:
            return None
        base_counts, _ = np.histogram(baseline.dropna(), bins=bins)
        curr_counts, _ = np.histogram(current.dropna(), bins=bins)
        base_counts = base_counts / max(1, base_counts.sum())
        curr_counts = curr_counts / max(1, curr_counts.sum())

    p = np.maximum(base_counts, EPSILON)
    q = np.maximum(curr_counts, EPSILON)
    psi = np.sum((p - q) * np.log(p / q))
    return float(psi)


def psi_flag(psi: float | None) -> str:
    if psi is None:
        return "OK"
    if psi < PSI_THRESHOLDS["OK"]:
        return "OK"
    if psi < PSI_THRESHOLDS["WARN"]:
        return "WARN"
    return "ACTION"


def ks_flag(stat: float | None) -> str:
    if stat is None:
        return "OK"
    if stat < KS_THRESHOLDS["OK"]:
        return "OK"
    if stat < KS_THRESHOLDS["WARN"]:
        return "WARN"
    return "ACTION"


def upsert_feature_drift(
    conn: sqlite3.Connection,
    week_start: str,
    ticker: str,
    feature_rows: Iterable[tuple[str, float | None, str]],
):
    for feature, psi, flag in feature_rows:
        conn.execute(
            "DELETE FROM drift_feature_weekly WHERE week_start=? AND ticker=? AND feature=?",
            (week_start, ticker, feature),
        )
        conn.execute(
            "INSERT INTO drift_feature_weekly (week_start, ticker, feature, psi, flag) VALUES (?,?,?,?,?)",
            (week_start, ticker, feature, psi, flag),
        )


def drift_feature_rows(
    df: pd.DataFrame,
    features: Sequence[str],
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
) -> list[tuple[str, float | None, str]]:
    rows = []
    base = df.loc[(df.index >= baseline_start) & (df.index <= baseline_end)]
    curr = df.loc[(df.index >= current_start) & (df.index <= current_end)]
    if base.empty or curr.empty:
        return rows
    for feat in features:
        if feat not in base.columns or feat not in curr.columns:
            continue
        psi = compute_psi(base[feat], curr[feat])
        rows.append((feat, psi, psi_flag(psi)))
    return rows


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def read_predictions_log(
    conn: sqlite3.Connection,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    model_version: str | None,
    features_checksum: str | None,
) -> list[float]:
    if not table_exists(conn, "predictions_log"):
        return []
    params = [ticker, start.isoformat(), end.isoformat()]
    where = "ticker=? AND ts>=? AND ts<=?"
    if model_version:
        where += " AND model_version=?"
        params.append(model_version)
    if features_checksum:
        where += " AND features_checksum=?"
        params.append(features_checksum)
    cur = conn.execute(f"SELECT proba FROM predictions_log WHERE {where}", params)
    return [r[0] for r in cur.fetchall()]


def load_model(cfg: TrainConfig, ticker: str, model_version: str) -> XGBClassifier:
    path = os.path.join(cfg.MODEL_DIR, f"{ticker}_{cfg.BAR_SIZE.replace(' ', '')}_{model_version}.json")
    model = XGBClassifier()
    model.load_model(path)
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    return model


def compute_predictions(
    model: XGBClassifier,
    feature_df: pd.DataFrame,
    features: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[float]:
    window = feature_df.loc[(feature_df.index >= start) & (feature_df.index <= end)]
    if window.empty:
        return []
    if not set(features).issubset(window.columns):
        return []
    window = window.dropna(subset=features)
    if window.empty:
        return []
    X = window.loc[:, features]
    proba = model.predict_proba(X)[:, 1]
    return proba.tolist()


def upsert_pred_drift(
    conn: sqlite3.Connection,
    week_start: str,
    ticker: str,
    proba_mean_shift: float | None,
    proba_std_shift: float | None,
    ks_stat: float | None,
    flag: str,
):
    conn.execute(
        "DELETE FROM drift_pred_weekly WHERE week_start=? AND ticker=?",
        (week_start, ticker),
    )
    conn.execute(
        """
        INSERT INTO drift_pred_weekly
        (week_start, ticker, proba_mean_shift, proba_std_shift, ks_stat, flag)
        VALUES (?,?,?,?,?,?)
        """,
        (week_start, ticker, proba_mean_shift, proba_std_shift, ks_stat, flag),
    )


def ks_statistic(baseline: Sequence[float], current: Sequence[float]) -> float | None:
    if len(baseline) == 0 or len(current) == 0:
        return None
    stat = ks_2samp(baseline, current, alternative="two-sided").statistic
    return float(stat)


def week_severity(conn: sqlite3.Connection, ticker: str, week_start: str) -> int:
    ks_flag_row = conn.execute(
        "SELECT flag FROM drift_pred_weekly WHERE ticker=? AND week_start=?",
        (ticker, week_start),
    ).fetchone()
    ks_flag_val = ks_flag_row[0] if ks_flag_row else "OK"

    psi_flags = conn.execute(
        "SELECT flag FROM drift_feature_weekly WHERE ticker=? AND week_start=?",
        (ticker, week_start),
    ).fetchall()
    psi_flag_vals = [r[0] for r in psi_flags] if psi_flags else []

    values = [FLAG_SEVERITY.get(ks_flag_val, 0)] + [FLAG_SEVERITY.get(f, 0) for f in psi_flag_vals]
    return max(values) if values else 0


def record_action(
    conn: sqlite3.Connection,
    ticker: str,
    level: str,
    action: str,
    reason: str,
    details: dict,
):
    conn.execute(
        """
        INSERT INTO actions_log (ts_utc, ticker, level, action, reason, details_json)
        VALUES (?,?,?,?,?,?)
        """,
        (
            datetime.utcnow().isoformat(),
            ticker,
            level,
            action,
            reason,
            json.dumps(details, default=str),
        ),
    )


def evaluate_actions(
    conn: sqlite3.Connection,
    ticker: str,
    week_start: str,
    psi_rows: list[tuple[str, float | None, str]],
    ks_flag_val: str,
    ks_stat_val: float | None,
):
    psi_action_feats = [feat for feat, _, flag in psi_rows if flag == "ACTION"]
    psi_action_trigger = len(psi_action_feats) >= 3
    ks_action_trigger = ks_flag_val == "ACTION"

    current_sev = max([FLAG_SEVERITY.get(ks_flag_val, 0)] + [FLAG_SEVERITY.get(f, 0) for _, _, f in psi_rows])
    prev_week = (pd.to_datetime(week_start) - pd.Timedelta(days=7)).date().isoformat()
    prev_sev = week_severity(conn, ticker, prev_week)
    consecutive_warn = current_sev == 1 and prev_sev == 1

    if psi_action_trigger or ks_action_trigger or consecutive_warn:
        reasons = []
        details = {
            "psi_action_features": psi_action_feats,
            "ks_stat": ks_stat_val,
            "ks_flag": ks_flag_val,
            "consecutive_warn": consecutive_warn,
        }
        if psi_action_trigger:
            reasons.append(f"PSI ACTION on {len(psi_action_feats)} features")
        if ks_action_trigger:
            reasons.append(f"KS ACTION (stat={ks_stat_val})")
        if consecutive_warn:
            reasons.append("WARN for 2 consecutive weeks")
        record_action(
            conn,
            ticker,
            "ACTION",
            "RETRAIN",
            f"; ".join(reasons) + f" @week_start={week_start}",
            details,
        )


def process_ticker(conn: sqlite3.Connection, cfg: TrainConfig, ticker: str, week_start: date, week_end: date):
    reg = latest_registry(conn, ticker)
    if not reg:
        print(f"{ticker}: no registry row found; skipping")
        return

    features = load_feature_list(cfg, ticker, cfg.SHAP_TOP_K)
    if not features:
        print(f"{ticker}: feature list missing; skipping")
        return

    bars = load_bars(conn, ticker)
    if bars.empty:
        print(f"{ticker}: no bars cached; skipping")
        return

    df_feat = build_feature_frame(bars, cfg)
    train_start = pd.to_datetime(reg["train_start"])
    train_end = pd.to_datetime(reg["train_end"])
    baseline_end = train_end
    baseline_start = max(train_start, baseline_end - pd.Timedelta(days=30))

    current_start = pd.to_datetime(week_start)
    current_end = pd.to_datetime(week_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    psi_rows = drift_feature_rows(
        df_feat,
        features,
        baseline_start,
        baseline_end,
        current_start,
        current_end,
    )
    ws_str = week_start.isoformat()
    conn.execute(
        "DELETE FROM drift_feature_weekly WHERE week_start=? AND ticker=?",
        (ws_str, ticker),
    )
    upsert_feature_drift(conn, ws_str, ticker, psi_rows)

    model = load_model(cfg, ticker, reg["model_version"])

    baseline_probs = read_predictions_log(conn, ticker, baseline_start, baseline_end, reg["model_version"], reg["features_checksum"])
    if not baseline_probs:
        baseline_probs = compute_predictions(model, df_feat, features, baseline_start, baseline_end)

    current_probs = read_predictions_log(conn, ticker, current_start, current_end, reg["model_version"], reg["features_checksum"])
    if not current_probs:
        current_probs = compute_predictions(model, df_feat, features, current_start, current_end)

    ks_stat_val = ks_statistic(baseline_probs, current_probs)
    flag = ks_flag(ks_stat_val)
    mean_shift = (np.mean(current_probs) - np.mean(baseline_probs)) if baseline_probs and current_probs else None
    std_shift = (np.std(current_probs) - np.std(baseline_probs)) if baseline_probs and current_probs else None
    upsert_pred_drift(conn, ws_str, ticker, mean_shift, std_shift, ks_stat_val, flag)

    evaluate_actions(conn, ticker, ws_str, psi_rows, flag, ks_stat_val)

    print(
        f"{ticker}: week_start={ws_str} psi_rows={len(psi_rows)} ks_stat={ks_stat_val} flag={flag}"
    )


def main():
    cfg = TrainConfig()
    ensure_schema(cfg.DATA_DB)

    week_start, week_end = completed_week_bounds()
    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        for t in cfg.TICKERS:
            try:
                process_ticker(conn, cfg, t, week_start, week_end)
            except Exception as e:
                print(f"{t}: drift monitor error: {e}")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
