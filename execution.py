from __future__ import annotations
import math
import pandas as pd
from config import BotConfig

def initial_stop_from_atr(entry: float, df_feat: pd.DataFrame, cfg: BotConfig) -> float:
    atr = float(df_feat["atr_14"].iloc[-1])
    return entry - (atr * cfg.ATR_MULT_STOP)

def calc_position_size(balance: float, entry: float, stop: float, cfg: BotConfig) -> int:
    risk_amount = balance * cfg.RISK_PER_TRADE
    risk_per_share = max(1e-6, abs(entry - stop))
    size_risk = int(risk_amount / risk_per_share)

    max_alloc = balance * cfg.ALLOC_PER_SYMBOL
    size_alloc = int(max_alloc / max(1e-6, entry))

    return max(0, min(size_risk, size_alloc))
