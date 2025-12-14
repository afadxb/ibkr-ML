# dashboard_monitor.py
import json
import sqlite3
import pandas as pd
import streamlit as st

from config import TrainConfig

cfg = TrainConfig()

@st.cache_data(ttl=60)
def load_table(query: str, params=()):
    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()
    return df

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.title("ML Monitoring Dashboard (SQLite)")

page = st.sidebar.radio("Page", ["Overview", "Ticker Drilldown", "Model Registry"])

# -------- Overview --------
if page == "Overview":
    days = st.sidebar.slider("Days", 7, 120, 30)
    df = load_table(
        "SELECT * FROM monitoring_metrics_daily WHERE date >= date('now', ?) ORDER BY date ASC",
        (f"-{days} day",)
    )

    if df.empty:
        st.warning("No monitoring data found. Run monitor.py first.")
        st.stop()

    latest = df.sort_values("date").groupby("ticker").tail(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers", latest["ticker"].nunique())
    c2.metric("Avg Missing Bars % (30d)", round(latest["missing_bars_pct"].mean(), 2))
    c3.metric("Avg Expectancy (net, holdout proxy)", round(latest["expectancy_net"].dropna().mean(), 6) if latest["expectancy_net"].notna().any() else "n/a")
    c4.metric("WARN/ACTION Count", int((latest["drift_flag"] != "OK").sum()))

    st.subheader("Latest Status")
    st.dataframe(latest.sort_values(["drift_flag","ticker"]), use_container_width=True)

    st.subheader("Missing Bars % (trend)")
    pivot = df.pivot_table(index="date", columns="ticker", values="missing_bars_pct", aggfunc="mean")
    st.line_chart(pivot)

    st.subheader("Expectancy Net (trend) [holdout proxy]")
    pivot2 = df.pivot_table(index="date", columns="ticker", values="expectancy_net", aggfunc="mean")
    st.line_chart(pivot2)

# -------- Drilldown --------
elif page == "Ticker Drilldown":
    tickers = list(cfg.TICKERS)
    t = st.sidebar.selectbox("Ticker", tickers)
    days = st.sidebar.slider("Days", 7, 180, 60)

    df = load_table(
        "SELECT * FROM monitoring_metrics_daily WHERE ticker=? AND date >= date('now', ?) ORDER BY date ASC",
        (t, f"-{days} day")
    )
    if df.empty:
        st.warning("No data for this ticker.")
        st.stop()

    top = df.tail(1).iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Drift Flag", top["drift_flag"])
    c2.metric("Missing Bars %", round(top["missing_bars_pct"], 2))
    c3.metric("Outlier Bars %", round(top["outlier_bars_pct"], 2))
    c4.metric("Holdout PF (proxy)", round(top["profit_factor_net"], 3) if pd.notna(top["profit_factor_net"]) else "n/a")
    c5.metric("Holdout Expectancy (proxy)", round(top["expectancy_net"], 6) if pd.notna(top["expectancy_net"]) else "n/a")

    st.subheader("Daily Metrics")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Bars %")
    st.line_chart(df.set_index("date")["missing_bars_pct"])

    st.subheader("Outlier Bars %")
    st.line_chart(df.set_index("date")["outlier_bars_pct"])

    st.subheader("Expectancy Net (holdout proxy)")
    st.line_chart(df.set_index("date")["expectancy_net"])

# -------- Registry --------
else:
    df = load_table("SELECT * FROM model_registry ORDER BY trained_at_utc DESC LIMIT 500")
    if df.empty:
        st.warning("No models in registry yet. Run trainer_wfo.py first.")
        st.stop()

    st.subheader("Model Registry (latest 500)")
    st.dataframe(df, use_container_width=True)

    t = st.selectbox("Compare ticker", list(cfg.TICKERS))
    df_t = df[df["ticker"] == t].copy()
    st.subheader(f"Recent versions: {t}")
    st.dataframe(df_t.head(20), use_container_width=True)

    if not df_t.empty:
        row = df_t.iloc[0]
        st.subheader("Latest Holdout Metrics")
        try:
            st.json(json.loads(row["holdout_metrics_json"]))
        except Exception:
            st.write(row["holdout_metrics_json"])

        st.subheader("Latest WFO Summary")
        try:
            st.json(json.loads(row["wfo_metrics_json"]))
        except Exception:
            st.write(row["wfo_metrics_json"])
