# trainer_wfo.py
import os
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from ib_insync import IB

from config import TrainConfig
from db_schema import ensure_schema
from ibkr_data import fetch_and_cache
from features import add_features, add_label
from metrics import trading_metrics, constrained_objective_foldscore

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def features_checksum(features: list[str]) -> str:
    return sha256_text("\n".join(features))

def shap_select_features(model: XGBClassifier, X: pd.DataFrame, top_k: int) -> list[str]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        Xs = X.sample(min(2000, len(X)), random_state=42) if len(X) > 2000 else X
        shap_vals = explainer.shap_values(Xs)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        imp = np.mean(np.abs(shap_vals), axis=0)
        order = np.argsort(imp)[::-1]
        return [X.columns[i] for i in order[:top_k]]
    except Exception:
        booster = model.get_booster()
        names = booster.feature_names or list(X.columns)
        gains = {n: booster.get_score(importance_type="gain").get(n, 0.0) for n in names}
        return sorted(gains, key=gains.get, reverse=True)[:top_k]

def compute_spw_baseline(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, min(20.0, neg / pos))

def make_xgb_classifier(params: dict) -> XGBClassifier:
    """
    Helper to build classifiers with a defined estimator type.
    xgboost>=3.x no longer sets `_estimator_type`, but sklearn helpers (and save_model)
    still expect it to exist.
    """
    model = XGBClassifier(**params)
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    return model

def optuna_objective(trial, X, y, fwd_ret, cfg: TrainConfig):
    # baseline spw from full training set (still OK; fold uses train-only baseline below)
    spw_base = compute_spw_baseline(y)
    spw = trial.suggest_float("scale_pos_weight", 0.7 * spw_base, 1.7 * spw_base)
    spw = max(1.0, min(20.0, spw))

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 900),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.7),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "scale_pos_weight": spw,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    tscv = TimeSeriesSplit(n_splits=cfg.N_SPLITS)
    fold_scores = []
    fold_primary = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        fr_va = fwd_ret.iloc[val_idx]

        # fold-corrected spw baseline (institution practice)
        spw_fold = compute_spw_baseline(y_tr)
        params_fold = dict(params)
        # keep trial’s spw near baseline but anchor to fold baseline by scaling
        params_fold["scale_pos_weight"] = max(1.0, min(20.0, params["scale_pos_weight"] * (spw_fold / spw_base)))

        m = make_xgb_classifier(params_fold)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        proba = m.predict_proba(X_va)[:, 1]
        tm = trading_metrics(y_va.values, proba, fr_va.values, cfg.PROBA_SIGNAL_TH, cfg.COST_BPS)
        score = constrained_objective_foldscore(tm, cfg.MIN_SIGNALS, cfg.MIN_PRECISION)
        fold_scores.append(score)
        fold_primary.append(tm["avg_fwd_ret_on_signals_net"])

    # robust aggregation: median primary, stability penalty via IQR
    primary = np.nanmedian(fold_primary)
    iqr = np.nanpercentile(fold_primary, 75) - np.nanpercentile(fold_primary, 25)
    penalty = 0.5 * (0.0 if np.isnan(iqr) else iqr)
    final = float(primary - penalty)

    return final

def wfo_splits(df: pd.DataFrame, cfg: TrainConfig):
    # uses datetime index; creates rolling windows
    start = df.index.min()
    end = df.index.max()
    # first train end = start + train months
    train_start = start
    train_end = train_start + relativedelta(months=cfg.WFO_TRAIN_MONTHS)
    while True:
        val_start = train_end
        val_end = val_start + relativedelta(months=cfg.WFO_VAL_MONTHS)
        if val_end >= end:
            break
        yield (train_start, train_end, val_start, val_end)
        # step forward
        train_start = train_start + relativedelta(months=cfg.WFO_STEP_MONTHS)
        train_end = train_start + relativedelta(months=cfg.WFO_TRAIN_MONTHS)

def evaluate_window(X_tr, y_tr, fr_tr, X_va, y_va, fr_va, best_params, cfg: TrainConfig):
    m = make_xgb_classifier(best_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    proba = m.predict_proba(X_va)[:, 1]
    return trading_metrics(y_va.values, proba, fr_va.values, cfg.PROBA_SIGNAL_TH, cfg.COST_BPS)

def split_holdout(df: pd.DataFrame, cfg: TrainConfig):
    n = len(df)
    cut = int(n * (1.0 - cfg.HOLDOUT_PCT))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def persist_registry(db_path: str, row: dict):
    conn = sqlite3.connect(db_path)
    try:
        cols = ",".join(row.keys())
        qs = ",".join(["?"] * len(row))
        conn.execute(f"INSERT INTO model_registry ({cols}) VALUES ({qs})", list(row.values()))
        conn.commit()
    finally:
        conn.close()

def train_one_ticker(ib: IB, ticker: str, cfg: TrainConfig):
    df_raw = fetch_and_cache(
        ib=ib,
        db_path=cfg.DATA_DB,
        ticker=ticker,
        bar_size=cfg.BAR_SIZE,
        what_to_show=cfg.WHAT_TO_SHOW,
        use_rth=cfg.USE_RTH,
    )

    df = add_features(df_raw, cfg.REGIME_ROLL)
    df = add_label(df, cfg.LABEL_HORIZON_BARS, cfg.LABEL_ATR_K, cfg.MIN_LABEL_PCT)
    df.dropna(inplace=True)

    # Avoid label leakage at tail
    df = df.iloc[:-cfg.LABEL_HORIZON_BARS].copy()

    # Features
    exclude = {"direction","label_thr","fwd_ret","Open","High","Low","Close","Volume"}
    features = [c for c in df.columns if c not in exclude]

    # Split holdout
    df_train, df_hold = split_holdout(df, cfg)

    X = df_train[features]
    y = df_train["direction"]
    fr = df_train["fwd_ret"]

    # Tune
    if cfg.FAST_MODE:
        best_params = {
            "n_estimators": 650,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
            "scale_pos_weight": compute_spw_baseline(y),
            "random_state": 42,
            "n_jobs": -1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        tuned_score = None
    else:
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: optuna_objective(t, X, y, fr, cfg), n_trials=cfg.N_TRIALS, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_params.update({
            "random_state": 42,
            "n_jobs": -1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
        tuned_score = float(study.best_value)

    # Baseline model for SHAP prune
    base = make_xgb_classifier(best_params)
    base.fit(X, y, verbose=False)
    pruned = shap_select_features(base, X, cfg.SHAP_TOP_K)

    # Final model training on pruned features
    Xp = df_train[pruned]
    yp = df_train["direction"]
    frp = df_train["fwd_ret"]

    final_model = make_xgb_classifier(best_params)
    final_model.fit(Xp, yp, verbose=False)

    # Holdout evaluation (headline)
    hold_metrics = {}
    if len(df_hold) > 50:
        Xh = df_hold[pruned]
        yh = df_hold["direction"]
        frh = df_hold["fwd_ret"]
        proba = final_model.predict_proba(Xh)[:, 1]
        hold_metrics = trading_metrics(yh.values, proba, frh.values, cfg.PROBA_SIGNAL_TH, cfg.COST_BPS)

    # WFO evaluation (fixed best params; no re-tune per window by default)
    wfo_rows = []
    for (tr_s, tr_e, va_s, va_e) in wfo_splits(df_train, cfg):
        dtr = df_train.loc[(df_train.index >= tr_s) & (df_train.index < tr_e)]
        dva = df_train.loc[(df_train.index >= va_s) & (df_train.index < va_e)]
        if len(dtr) < 300 or len(dva) < 50:
            continue

        X_tr = dtr[pruned]; y_tr = dtr["direction"]; fr_tr = dtr["fwd_ret"]
        X_va = dva[pruned]; y_va = dva["direction"]; fr_va = dva["fwd_ret"]

        tm = evaluate_window(X_tr, y_tr, fr_tr, X_va, y_va, fr_va, best_params, cfg)
        tm.update({
            "train_start": tr_s.isoformat(),
            "train_end": tr_e.isoformat(),
            "val_start": va_s.isoformat(),
            "val_end": va_e.isoformat(),
        })
        wfo_rows.append(tm)

    wfo_summary = {}
    if wfo_rows:
        dfw = pd.DataFrame(wfo_rows)
        wfo_summary = {
            "windows": len(dfw),
            "median_expectancy_net": float(np.nanmedian(dfw["expectancy_net"])),
            "median_pf_net": float(np.nanmedian(dfw["profit_factor_net"])),
            "median_precision": float(np.nanmedian(dfw["precision"])),
            "median_signals": float(np.nanmedian(dfw["signals"])),
        }

    # Schema persistence
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_{cfg.BAR_SIZE.replace(' ','')}_{version}.json")
    feats_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_features.txt")
    meta_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_meta.json")

    final_model.save_model(model_path)

    chk = features_checksum(pruned)
    with open(feats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pruned))

    meta = {
        "ticker": ticker,
        "model_version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "bar_size": cfg.BAR_SIZE,
        "label": {
            "horizon_hours": cfg.LABEL_HORIZON_HOURS,
            "horizon_bars": cfg.LABEL_HORIZON_BARS,
            "atr_k": cfg.LABEL_ATR_K,
            "min_label_pct": cfg.MIN_LABEL_PCT,
        },
        "objective": {
            "primary": "median(avg_fwd_ret_on_signals_net) - 0.5*IQR",
            "min_signals": cfg.MIN_SIGNALS,
            "min_precision": cfg.MIN_PRECISION,
            "proba_signal_th": cfg.PROBA_SIGNAL_TH,
            "cost_bps": cfg.COST_BPS,
        },
        "best_params": best_params,
        "tuned_score": tuned_score,
        "features_checksum": chk,
        "features_count": len(pruned),
        "holdout_metrics": hold_metrics,
        "wfo_summary": wfo_summary,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Registry row
    row = {
        "ticker": ticker,
        "model_version": version,
        "trained_at_utc": meta["trained_at_utc"],
        "train_start": df_train.index.min().isoformat(),
        "train_end": df_train.index.max().isoformat(),
        "bar_size": cfg.BAR_SIZE,
        "label_horizon_hours": cfg.LABEL_HORIZON_HOURS,
        "label_params_json": json.dumps(meta["label"]),
        "best_params_json": json.dumps(best_params),
        "features_checksum": chk,
        "features_count": len(pruned),
        "holdout_metrics_json": json.dumps(hold_metrics),
        "wfo_metrics_json": json.dumps(wfo_summary),
    }
    persist_registry(cfg.DATA_DB, row)

    return {
        "ticker": ticker,
        "model_path": model_path,
        "features_count": len(pruned),
        "holdout": hold_metrics,
        "wfo": wfo_summary,
    }

def main():
    cfg = TrainConfig()
    ensure_schema(cfg.DATA_DB)

    ib = IB()
    ib.connect(cfg.IB_HOST, cfg.IB_PORT, clientId=cfg.IB_CLIENT_ID)

    results = []
    try:
        for t in cfg.TICKERS:
            print(f"\n=== TRAIN {t} ({cfg.BAR_SIZE}) ===")
            res = train_one_ticker(ib, t, cfg)
            results.append(res)
            print(f"{t}: features={res['features_count']} holdout={res['holdout']} wfo={res['wfo']}")
    finally:
        ib.disconnect()

    print("\nDONE")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()

