from __future__ import annotations
import pandas as pd
from config import BotConfig

def entry_threshold(df_feat: pd.DataFrame, cfg: BotConfig) -> float:
    th = cfg.PROBA_SIGNAL_TH
    if cfg.USE_VOLATILITY_GATE and "regime_high_vol" in df_feat.columns:
        hv = int(df_feat["regime_high_vol"].iloc[-1])
        if hv == 1:
            th = min(0.80, th + cfg.HIGH_VOL_TH_BUMP)
    return th

def allow_long(df_feat: pd.DataFrame, cfg: BotConfig) -> bool:
    if cfg.USE_SUPERTREND_GATE and "st_direction" in df_feat.columns:
        return int(df_feat["st_direction"].iloc[-1]) == 1
    return True

def decide(prob_up: float, df_feat: pd.DataFrame, cfg: BotConfig) -> dict:
    th = entry_threshold(df_feat, cfg)
    gate = allow_long(df_feat, cfg)
    buy = gate and (prob_up >= th)
    return {
        "decision": "BUY" if buy else "HOLD",
        "used_threshold": float(th),
        "gate_pass": bool(gate),
        "st_direction": int(df_feat["st_direction"].iloc[-1]) if "st_direction" in df_feat.columns else None,
        "regime_high_vol": int(df_feat["regime_high_vol"].iloc[-1]) if "regime_high_vol" in df_feat.columns else None,
    }
