import pandas as pd

from config import BotConfig
from features import add_features, add_label
from supertrend_params import normalize_timeframe_label


def make_features(
    df_bars: pd.DataFrame,
    cfg: BotConfig,
    symbol: str,
    timeframe: str | None = None,
    st_params: dict | tuple | None = None,
) -> pd.DataFrame:
    tf = timeframe or normalize_timeframe_label(cfg.BAR_SIZE)
    return add_features(
        df_bars,
        regime_roll=cfg.REGIME_ROLL,
        st_period=cfg.SUPERTREND_PERIOD,
        st_multiplier=cfg.SUPERTREND_MULTIPLIER,
        symbol=symbol,
        timeframe=tf,
        db_path=cfg.DATA_DB,
        st_params=st_params,
    )


def make_labeled_frame(
    df_bars: pd.DataFrame,
    cfg: BotConfig,
    symbol: str,
    timeframe: str | None = None,
    st_params: dict | tuple | None = None,
) -> pd.DataFrame:
    df = make_features(df_bars, cfg, symbol=symbol, timeframe=timeframe, st_params=st_params)
    df = add_label(df, cfg.LABEL_HORIZON_BARS, cfg.LABEL_ATR_K, cfg.MIN_LABEL_PCT)
    return df
