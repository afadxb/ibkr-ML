import pandas as pd
from config import BotConfig
from features import add_features, add_label

def make_features(df_bars: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    return add_features(
        df_bars,
        regime_roll=cfg.REGIME_ROLL,
        st_period=cfg.SUPERTREND_PERIOD,
        st_multiplier=cfg.SUPERTREND_MULTIPLIER,
    )

def make_labeled_frame(df_bars: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    df = make_features(df_bars, cfg)
    df = add_label(df, cfg.LABEL_HORIZON_BARS, cfg.LABEL_ATR_K, cfg.MIN_LABEL_PCT)
    return df
