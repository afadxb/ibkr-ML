from __future__ import annotations
import os, json, sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from ib_insync import IB

from config import BotConfig
from db_schema import ensure_schema
from datahub import refresh_cache
from pipeline import make_labeled_frame
from metrics import trading_metrics, constrained_fold_score
from utils import features_checksum, utc_now_iso
from cv import timestamp_folds
from supertrend_params import (
    normalize_timeframe_label,
    select_supertrend_params,
    save_st_recommendations,
)

def compute_spw_baseline(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, min(20.0, neg / pos))

def make_xgb_classifier(params: dict) -> XGBClassifier:
    """
    Ensure sklearn wrapper carries estimator type so save_model works with xgboost>=3.
    """
    m = XGBClassifier(**params)
    if not hasattr(m, "_estimator_type"):
        m._estimator_type = "classifier"
    return m

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

def optuna_objective(trial, X, y, fwd_ret, df_index, cfg: BotConfig):
    spw_base = compute_spw_baseline(y)
    spw = trial.suggest_float("scale_pos_weight", 0.7*spw_base, 1.7*spw_base)
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

    folds = timestamp_folds(
        df_index, n_folds=cfg.N_SPLITS,
        val_days=cfg.CV_VAL_DAYS,
        gap_days=cfg.CV_GAP_DAYS,
        min_train_bars=cfg.CV_MIN_TRAIN_BARS,
        mode=cfg.CV_MODE,
        rolling_train_months=cfg.CV_ROLLING_TRAIN_MONTHS
    )
    if not folds:
        return -999.0

    fold_primary = []
    for f in folds:
        tr_mask = (df_index >= f.train_start) & (df_index < f.train_end)
        va_mask = (df_index >= f.val_start) & (df_index < f.val_end)

        X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
        X_va, y_va = X.loc[va_mask], y.loc[va_mask]
        fr_va = fwd_ret.loc[va_mask]

        spw_fold = compute_spw_baseline(y_tr)
        params_fold = dict(params)
        params_fold["scale_pos_weight"] = max(1.0, min(20.0, params["scale_pos_weight"] * (spw_fold / spw_base)))

        m = make_xgb_classifier(params_fold)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = m.predict_proba(X_va)[:, 1]

        tm = trading_metrics(y_va.values, proba, fr_va.values, cfg.PROBA_SIGNAL_TH, cfg.COST_BPS)
        fold_primary.append(tm["avg_fwd_ret_on_signals_net"])

    primary = np.nanmedian(fold_primary)
    iqr = np.nanpercentile(fold_primary, 75) - np.nanpercentile(fold_primary, 25)
    penalty = 0.5 * (0.0 if np.isnan(iqr) else iqr)
    return float(primary - penalty)

def split_holdout(df: pd.DataFrame, cfg: BotConfig):
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

def train_one(ib: IB, ticker: str, cfg: BotConfig) -> dict:
    bars = refresh_cache(ib, cfg.DATA_DB, ticker, cfg.BAR_SIZE, cfg.WHAT_TO_SHOW, cfg.USE_RTH)
    timeframe = normalize_timeframe_label(cfg.BAR_SIZE)
    st_row = select_supertrend_params(bars, cost_bps=6.0, timeframe=timeframe, symbol=ticker)
    st_params: tuple[int, float, dict | None]
    if st_row:
        print(f"{ticker}: selected ST params atr_len={st_row.atr_len} mult={st_row.mult} score={st_row.score:.4f}")
        save_st_recommendations(cfg.DATA_DB, [st_row])
        st_params = (st_row.atr_len, st_row.mult, {"score": st_row.score, "as_of": st_row.as_of, "run_id": st_row.run_id})

        artifact = {
            "symbol": st_row.symbol,
            "timeframe": st_row.timeframe,
            "atr_len": st_row.atr_len,
            "mult": st_row.mult,
            "score": st_row.score,
            "as_of": st_row.as_of,
            "run_id": st_row.run_id,
        }
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        safe_run_id = st_row.run_id.replace(":", "").replace("-", "")
        art_path = os.path.join(cfg.MODEL_DIR, f"st_params_{timeframe}_{safe_run_id}.json")
        with open(art_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
    else:
        print(f"{ticker}: using default ST params atr_len={cfg.SUPERTREND_PERIOD} mult={cfg.SUPERTREND_MULTIPLIER}")
        st_params = (cfg.SUPERTREND_PERIOD, cfg.SUPERTREND_MULTIPLIER, None)

    df = make_labeled_frame(bars, cfg, symbol=ticker, timeframe=timeframe, st_params=st_params)
    df.dropna(inplace=True)

    # avoid label leakage tail
    df = df.iloc[:-cfg.LABEL_HORIZON_BARS].copy()

    # exclude target + price/volume columns that are not model inputs
    exclude = {"direction","Open","High","Low","Close","Volume"}
    features = [c for c in df.columns if c not in exclude]

    df_train, df_hold = split_holdout(df, cfg)

    X = df_train[features]
    y = df_train["direction"]
    fr = df_train["fwd_ret"]
    idx = df_train.index

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
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda t: optuna_objective(t, X, y, fr, idx, cfg), n_trials=cfg.N_TRIALS, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_params.update({
            "random_state": 42,
            "n_jobs": -1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
        tuned_score = float(study.best_value)

    base = make_xgb_classifier(best_params)
    base.fit(X, y, verbose=False)
    pruned = shap_select_features(base, X, cfg.SHAP_TOP_K)

    Xp = df_train[pruned]
    yp = df_train["direction"]
    frp = df_train["fwd_ret"]

    final = make_xgb_classifier(best_params)
    final.fit(Xp, yp, verbose=False)

    hold_metrics = {}
    if len(df_hold) > 50:
        Xh = df_hold[pruned]
        yh = df_hold["direction"]
        frh = df_hold["fwd_ret"]
        proba = final.predict_proba(Xh)[:, 1]
        hold_metrics = trading_metrics(yh.values, proba, frh.values, cfg.PROBA_SIGNAL_TH, cfg.COST_BPS)

    # Persist artifacts
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_{cfg.BAR_SIZE.replace(' ','')}_{version}.json")
    feats_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_features.txt")
    meta_path = os.path.join(cfg.MODEL_DIR, f"{ticker}_meta.json")

    final.save_model(model_path)
    with open(feats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pruned))
    chk = features_checksum(pruned)

    cv_scheme = {
        "type": "timestamp_folds",
        "val_days": cfg.CV_VAL_DAYS,
        "gap_days": cfg.CV_GAP_DAYS,
        "min_train_bars": cfg.CV_MIN_TRAIN_BARS,
        "mode": cfg.CV_MODE,
        "rolling_train_months": cfg.CV_ROLLING_TRAIN_MONTHS,
    }

    meta = {
        "ticker": ticker,
        "model_version": version,
        "trained_at_utc": utc_now_iso(),
        "bar_size": cfg.BAR_SIZE,
        "label": {
            "horizon_hours": cfg.LABEL_HORIZON_HOURS,
            "horizon_bars": cfg.LABEL_HORIZON_BARS,
            "atr_k": cfg.LABEL_ATR_K,
            "min_label_pct": cfg.MIN_LABEL_PCT,
        },
        "feature_engineering": {
            "regime_roll": cfg.REGIME_ROLL,
            "supertrend_period": cfg.SUPERTREND_PERIOD,
            "supertrend_multiplier": cfg.SUPERTREND_MULTIPLIER,
        },
        "objective": {
            "proba_signal_th": cfg.PROBA_SIGNAL_TH,
            "min_signals": cfg.MIN_SIGNALS,
            "min_precision": cfg.MIN_PRECISION,
            "cost_bps": cfg.COST_BPS,
        },
        "cv_scheme": cv_scheme,
        "best_params": best_params,
        "tuned_score": tuned_score,
        "features_checksum": chk,
        "features_count": len(pruned),
        "holdout_metrics": hold_metrics,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    persist_registry(cfg.DATA_DB, {
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
        "wfo_metrics_json": json.dumps({}),
        "cv_scheme_json": json.dumps(cv_scheme),
    })

    return {"ticker": ticker, "model_path": model_path, "features": len(pruned), "holdout": hold_metrics}

def main():
    cfg = BotConfig()
    ensure_schema(cfg.DATA_DB)

    ib = IB()
    ib.connect(cfg.IB_HOST, cfg.IB_PORT, clientId=cfg.IB_CLIENT_ID_TRAIN)
    try:
        for t in cfg.TICKERS:
            print(f"=== TRAIN {t} ===")
            out = train_one(ib, t, cfg)
            print(out)
    finally:
        ib.disconnect()

if __name__ == "__main__":
    main()
