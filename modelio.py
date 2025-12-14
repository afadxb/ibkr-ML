from __future__ import annotations
import os, glob, sqlite3
import xgboost as xgb
import pandas as pd
from config import BotConfig
from utils import features_checksum

def load_features_list(ticker: str, cfg: BotConfig) -> list[str]:
    path = os.path.join(cfg.MODEL_DIR, f"{ticker}_features.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing features file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f if line.strip()]
    if not feats:
        raise ValueError(f"Empty features list: {path}")
    return feats

def latest_registry_row(ticker: str, cfg: BotConfig) -> dict | None:
    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        cur = conn.execute(
            "SELECT model_version, features_checksum FROM model_registry "
            "WHERE ticker=? ORDER BY id DESC LIMIT 1",
            (ticker,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"model_version": row[0], "features_checksum": row[1]}
    finally:
        conn.close()

def resolve_model_path(ticker: str, model_version: str | None, cfg: BotConfig) -> str:
    if model_version:
        pattern = os.path.join(cfg.MODEL_DIR, f"{ticker}_*_{model_version}.json")
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    matches = glob.glob(os.path.join(cfg.MODEL_DIR, f"{ticker}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No model files for {ticker} in {cfg.MODEL_DIR}")
    return sorted(matches)[-1]

def load_booster(model_path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(model_path)
    return b

def load_bundle(ticker: str, cfg: BotConfig) -> dict:
    feats = load_features_list(ticker, cfg)
    chk = features_checksum(feats)
    reg = latest_registry_row(ticker, cfg)
    mv = reg["model_version"] if reg else None
    model_path = resolve_model_path(ticker, mv, cfg)
    booster = load_booster(model_path)

    if reg and reg.get("features_checksum") and reg["features_checksum"] != chk:
        raise RuntimeError(
            f"{ticker}: FEATURE CHECKSUM MISMATCH (registry vs features.txt). Hard failing."
        )

    return {
        "ticker": ticker,
        "model_version": mv or "unknown",
        "model_path": model_path,
        "features": feats,
        "features_checksum": chk,
        "booster": booster,
    }

def build_inference_row(df_feat: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df_feat.columns]
    if missing:
        raise RuntimeError(f"Missing required features: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df_feat[required].iloc[[-1]].copy()
    if X.isna().any(axis=None):
        nan_cols = X.columns[X.isna().iloc[0]].tolist()
        raise RuntimeError(f"NaNs in required features on latest row: {nan_cols[:10]}{'...' if len(nan_cols)>10 else ''}")
    return X
