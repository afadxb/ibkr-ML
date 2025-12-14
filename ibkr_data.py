# ibkr_data.py
import sqlite3
import pandas as pd
from ib_insync import IB, Stock, util

def get_cached_data(db_path: str, ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, index_col="timestamp", parse_dates=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(None)
            df = df[~df.index.isna()]
            df.sort_index(inplace=True)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def save_cache(db_path: str, ticker: str, df: pd.DataFrame) -> None:
    conn = sqlite3.connect(db_path)
    try:
        # ensure sqlite-friendly index (naive datetime)
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(None)
        df.to_sql(ticker, conn, if_exists="replace", index_label="timestamp")
        conn.commit()
    finally:
        conn.close()

def _fetch_bars(ib: IB, contract, duration: str, bar_size: str, what_to_show: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1
    )
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df["timestamp"] = pd.to_datetime(df["date"])
    df.set_index("timestamp", inplace=True)
    df = df[["open","high","low","close","volume"]].rename(columns=str.capitalize)
    df.sort_index(inplace=True)
    return df

def fetch_and_cache(
    ib: IB,
    db_path: str,
    ticker: str,
    bar_size: str,
    what_to_show: str,
    use_rth: bool,
    duration_full: str = "3 Y",
    duration_refresh: str = "45 D"
) -> pd.DataFrame:
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)

    cached = get_cached_data(db_path, ticker)
    duration = duration_full if cached.empty else duration_refresh

    new_df = _fetch_bars(ib, contract, duration, bar_size, what_to_show, use_rth)
    if new_df.empty and not cached.empty:
        return cached
    if new_df.empty:
        raise RuntimeError(f"{ticker}: No bars returned from IBKR.")

    # optional implied vol
    iv_df = _fetch_bars(ib, contract, duration, bar_size, "OPTION_IMPLIED_VOLATILITY", use_rth)
    if not iv_df.empty:
        new_df["implied_vol"] = iv_df["Close"].reindex(new_df.index)

    # normalize indexes to datetime to avoid mixed Timestamp/str sorting errors
    new_df.index = pd.to_datetime(new_df.index, errors="coerce", utc=True).tz_convert(None)
    cached.index = pd.to_datetime(cached.index, errors="coerce", utc=True).tz_convert(None) if not cached.empty else cached.index

    df = pd.concat([cached, new_df]) if not cached.empty else new_df
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    save_cache(db_path, ticker, df)
    return df
