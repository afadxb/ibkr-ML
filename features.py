import numpy as np
import pandas as pd

from config import BotConfig

def true_range(high, low, close):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def supertrend(
    df: pd.DataFrame,
    period: int = BotConfig.SUPERTREND_PERIOD,
    multiplier: float = BotConfig.SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    df = df.copy()
    atr = true_range(df["High"], df["Low"], df["Close"]).rolling(period).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)

    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    st.iloc[:period] = np.nan
    direction.iloc[:period] = 1

    for i in range(period, len(df)):
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        if np.isnan(prev_st):
            prev_st = lower.iloc[i - 1]

        if prev_dir == 1:
            st.iloc[i] = max(lower.iloc[i], prev_st)
            direction.iloc[i] = -1 if df["Close"].iloc[i] < st.iloc[i] else 1
        else:
            st.iloc[i] = min(upper.iloc[i], prev_st)
            direction.iloc[i] = 1 if df["Close"].iloc[i] > st.iloc[i] else -1

    df["supertrend"] = st
    df["st_direction"] = direction

    # Supertrend-derived features (must be used consistently in training + live)
    df["dist_supertrend"] = df["Close"] / df["supertrend"] - 1
    df["st_flip"] = (df["st_direction"] != df["st_direction"].shift(1)).astype(int)
    df["bars_since_st_flip"] = df["st_flip"].groupby((df["st_flip"] == 1).cumsum()).cumcount()
    df["st_slope"] = df["supertrend"].pct_change(3)

    return df

def add_features(
    df: pd.DataFrame,
    regime_roll: int,
    st_period: int = BotConfig.SUPERTREND_PERIOD,
    st_multiplier: float = BotConfig.SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    df = df.copy()

    # Validate datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["hl_range"] = df["High"] - df["Low"]
    df["oc_range"] = (df["Close"] - df["Open"]).abs()
    df["body_ratio"] = df["oc_range"] / df["hl_range"].replace(0, np.nan)

    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["upper_shadow_pct"] = df["upper_shadow"] / df["hl_range"].replace(0, np.nan)
    df["lower_shadow_pct"] = df["lower_shadow"] / df["hl_range"].replace(0, np.nan)

    ema_periods = [8, 13, 21, 34, 55, 89]
    for p in ema_periods:
        ema = df["Close"].ewm(span=p, adjust=False).mean()
        df[f"ema_{p}"] = ema
        df[f"dist_ema_{p}"] = df["Close"] / ema - 1

    for p in [7, 9, 14]:
        df[f"rsi_{p}"] = rsi(df["Close"], p)

    ema_fast = df["Close"].ewm(span=8, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=17, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    for length, std in [(20, 2.0), (13, 1.8)]:
        mid = df["Close"].rolling(length).mean()
        sd = df["Close"].rolling(length).std()
        upper = mid + sd * std
        lower = mid - sd * std
        df[f"bb_upper_{length}"] = upper
        df[f"bb_lower_{length}"] = lower
        df[f"bb_width_{length}"] = (upper - lower) / mid.replace(0, np.nan)
        df[f"bb_pct_{length}"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)

    tr = true_range(df["High"], df["Low"], df["Close"])
    df["atr_14"] = tr.rolling(14).mean()
    df["atrp_14"] = df["atr_14"] / df["Close"].replace(0, np.nan)

    df["vol_sma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma20"].replace(0, np.nan)

    # Momentum + liquidity
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_6"] = df["Close"].pct_change(6)
    df["roc_20"] = df["Close"].pct_change(20)
    df["dollar_vol"] = df["Close"] * df["Volume"]
    df["dollar_vol_sma20"] = df["dollar_vol"].rolling(20).mean()
    df["rel_dollar_vol"] = df["dollar_vol"] / df["dollar_vol_sma20"].replace(0, np.nan)

    # Day-of-week as integer
    df["dow"] = df.index.dayofweek

    # Supertrend + derived features (single source of truth)
    df = supertrend(df, period=st_period, multiplier=st_multiplier)

    # Regime
    df["atrp_14_z"] = (df["atrp_14"] - df["atrp_14"].rolling(regime_roll).mean()) / df["atrp_14"].rolling(regime_roll).std()
    df["bb_width_20_z"] = (df["bb_width_20"] - df["bb_width_20"].rolling(regime_roll).mean()) / df["bb_width_20"].rolling(regime_roll).std()

    q75 = df["atrp_14"].rolling(regime_roll).quantile(0.75)
    q25 = df["atrp_14"].rolling(regime_roll).quantile(0.25)
    df["regime_high_vol"] = (df["atrp_14"] > q75).astype(int)
    df["regime_low_vol"] = (df["atrp_14"] < q25).astype(int)

    df["regime_trend_up"] = ((df["st_direction"] == 1) & (df["ema_21"] > df["ema_55"])).astype(int)
    df["regime_trend_down"] = ((df["st_direction"] == -1) & (df["ema_21"] < df["ema_55"])).astype(int)

    return df

def add_label(df: pd.DataFrame, horizon_bars: int, atr_k: float, min_label_pct: float) -> pd.DataFrame:
    df = df.copy()
    fwd_close = df["Close"].shift(-horizon_bars)
    df["fwd_ret"] = (fwd_close / df["Close"]) - 1.0

    thr = (atr_k * (df["atr_14"] / df["Close"].replace(0, np.nan))).fillna(min_label_pct)
    thr = np.maximum(thr, min_label_pct)
    df["label_thr"] = thr
    df["direction"] = (df["fwd_ret"] >= df["label_thr"]).astype(int)
    return df
