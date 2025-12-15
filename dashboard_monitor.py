import json
import sqlite3
import pandas as pd
import streamlit as st

from config import BotConfig
from supertrend_params import normalize_timeframe_label

cfg = BotConfig()

def _conn():
    return sqlite3.connect(cfg.DATA_DB)

@st.cache_data(ttl=60)
def load_df(query: str, params=()):
    conn = _conn()
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=60)
def load_latest_st_params():
    return load_df(
        """
        SELECT * FROM st_param_recommendations
        WHERE (symbol, timeframe, as_of) IN (
            SELECT symbol, timeframe, MAX(as_of) FROM st_param_recommendations GROUP BY symbol, timeframe
        )
        ORDER BY symbol ASC
        """
    )

st.set_page_config(page_title="SteadyAlpha ML Monitoring", layout="wide")
st.title("SteadyAlpha ML Monitoring (SQLite)")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Ticker Drilldown", "Predictions", "Drift Center", "Model Registry", "Supertrend Params"],
)

# ---------------- Overview ----------------
if page == "Overview":
    days = st.sidebar.slider("Days (daily monitoring)", 7, 180, 30)

    daily = load_df(
        "SELECT * FROM monitoring_metrics_daily WHERE date >= date('now', ?) ORDER BY date ASC",
        (f"-{days} day",),
    )

    pred7 = load_df(
        "SELECT ticker, prob_up, ts_utc, decision FROM predictions_log "
        "WHERE ts_utc >= datetime('now', '-7 day')"
    )
    drift_latest = load_df(
        "SELECT week_start, ticker, flag, ks_stat FROM drift_pred_weekly "
        "WHERE week_start = (SELECT MAX(week_start) FROM drift_pred_weekly)"
    )
    st_latest = load_latest_st_params()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers", len(cfg.TICKERS))

    if not daily.empty:
        latest = daily.sort_values("date").groupby("ticker").tail(1)
        c2.metric("Avg Missing Bars %", round(float(latest["missing_bars_pct"].mean()), 2))
        c3.metric("Avg Outlier Bars %", round(float(latest["outlier_bars_pct"].mean()), 2))
        c4.metric("WARN/ACTION (daily)", int((latest["drift_flag"] != "OK").sum()))
    else:
        c2.metric("Avg Missing Bars %", "n/a")
        c3.metric("Avg Outlier Bars %", "n/a")
        c4.metric("WARN/ACTION (daily)", "n/a")

    if not drift_latest.empty:
        c5.metric("WARN/ACTION (weekly KS)", int((drift_latest["flag"] != "OK").sum()))
    else:
        c5.metric("WARN/ACTION (weekly KS)", "n/a")

    st.subheader("Latest Daily Monitoring Status")
    if not daily.empty:
        latest = daily.sort_values("date").groupby("ticker").tail(1)
        st.dataframe(latest.sort_values(["drift_flag", "ticker"]), use_container_width=True)
    else:
        st.info("No monitoring_metrics_daily found. Run your daily monitor job if you want these tiles.")

    st.subheader("Predictions (last 7 days)")
    if pred7.empty:
        st.info("No predictions_log rows yet. Run live_bot.py to populate.")
    else:
        pred7["ts_utc"] = pd.to_datetime(pred7["ts_utc"], utc=True)
        buy_rate = (pred7["decision"] == "BUY").mean()
        st.metric("BUY rate (7d)", f"{buy_rate*100:.1f}%")
        # Mean proba by day
        pred7["day"] = pred7["ts_utc"].dt.date
        series = pred7.groupby(["day", "ticker"])["prob_up"].mean().unstack("ticker")
        st.line_chart(series)

    st.subheader("Supertrend params (latest)")
    if st_latest.empty:
        st.info("No supertrend recommendations saved yet. Train to populate st_param_recommendations.")
    else:
        cols = ["symbol", "timeframe", "atr_len", "mult", "score", "as_of", "run_id"]
        st.dataframe(st_latest[cols], use_container_width=True)

    st.subheader("Weekly KS Drift (latest week)")
    if drift_latest.empty:
        st.info("No drift_pred_weekly rows yet. Run drift_weekly.py.")
    else:
        st.dataframe(drift_latest.sort_values(["flag", "ks_stat"], ascending=[True, False]), use_container_width=True)

# ---------------- Ticker Drilldown ----------------
elif page == "Ticker Drilldown":
    t = st.sidebar.selectbox("Ticker", list(cfg.TICKERS))
    days = st.sidebar.slider("Days", 7, 365, 90)

    daily = load_df(
        "SELECT * FROM monitoring_metrics_daily WHERE ticker=? AND date >= date('now', ?) ORDER BY date ASC",
        (t, f"-{days} day"),
    )

    c1, c2, c3, c4 = st.columns(4)
    if daily.empty:
        c1.metric("Drift Flag", "n/a")
        c2.metric("Missing Bars %", "n/a")
        c3.metric("Outlier Bars %", "n/a")
        c4.metric("Notes", "n/a")
        st.info("No daily monitoring rows for this ticker.")
    else:
        top = daily.iloc[-1]
        c1.metric("Drift Flag", str(top["drift_flag"]))
        c2.metric("Missing Bars %", round(float(top["missing_bars_pct"]), 2))
        c3.metric("Outlier Bars %", round(float(top["outlier_bars_pct"]), 2))
        c4.metric("Notes", str(top.get("notes", ""))[:60])

        tf_label = normalize_timeframe_label(cfg.BAR_SIZE)
        st_row = load_df(
            "SELECT atr_len, mult, as_of, run_id, score FROM st_param_recommendations "
            "WHERE symbol=? AND timeframe=? ORDER BY as_of DESC LIMIT 1",
            (t, tf_label),
        )

        st.subheader("Supertrend params (latest)")
        if st_row.empty:
            st.info("No params saved yet for this ticker/timeframe.")
        else:
            r = st_row.iloc[0]
            cst1, cst2, cst3, cst4 = st.columns(4)
            cst1.metric("atr_len", int(r["atr_len"]))
            cst2.metric("mult", float(r["mult"]))
            cst3.metric("as_of", str(r["as_of"]))
            cst4.metric("run_id", str(r["run_id"]))
            st.markdown(f"Score: **{float(r['score']):.4f}**")

        st.subheader("Daily Monitoring Table")
        st.dataframe(daily, use_container_width=True)

        st.subheader("Missing Bars %")
        st.line_chart(daily.set_index("date")["missing_bars_pct"])

        st.subheader("Outlier Bars %")
        st.line_chart(daily.set_index("date")["outlier_bars_pct"])

    st.subheader("Weekly Drift (PSI + KS)")
    ks = load_df(
        "SELECT * FROM drift_pred_weekly WHERE ticker=? ORDER BY week_start DESC LIMIT 52",
        (t,),
    )
    psi = load_df(
        "SELECT * FROM drift_feature_weekly WHERE ticker=? ORDER BY week_start DESC, psi DESC LIMIT 500",
        (t,),
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**KS Drift (weekly)**")
        if ks.empty:
            st.info("No KS drift rows yet.")
        else:
            st.dataframe(ks, use_container_width=True)
            st.line_chart(ks.set_index("week_start")["ks_stat"])
    with colB:
        st.markdown("**Top PSI Features (latest weeks)**")
        if psi.empty:
            st.info("No PSI rows yet.")
        else:
            st.dataframe(psi, use_container_width=True)

# ---------------- Predictions ----------------
elif page == "Predictions":
    t = st.sidebar.selectbox("Ticker", list(cfg.TICKERS))
    days = st.sidebar.slider("Days", 1, 180, 30)

    pred = load_df(
        "SELECT ts_utc, prob_up, used_threshold, decision, st_direction, regime_high_vol "
        "FROM predictions_log WHERE ticker=? AND ts_utc >= datetime('now', ?) "
        "ORDER BY ts_utc ASC",
        (t, f"-{days} day"),
    )
    if pred.empty:
        st.info("No predictions for this ticker yet. Run live_bot.py.")
        st.stop()

    pred["ts_utc"] = pd.to_datetime(pred["ts_utc"], utc=True)
    pred["day"] = pred["ts_utc"].dt.date

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(pred))
    c2.metric("Mean p(up)", f"{pred['prob_up'].mean():.3f}")
    c3.metric("Std p(up)", f"{pred['prob_up'].std():.3f}")
    c4.metric("BUY rate", f"{(pred['decision']=='BUY').mean()*100:.1f}%")

    st.subheader("Probability time series")
    ts = pred.set_index("ts_utc")[["prob_up"]]
    st.line_chart(ts)

    st.subheader("Daily mean probability")
    daily_mean = pred.groupby("day")["prob_up"].mean()
    st.line_chart(daily_mean)

    st.subheader("Decision counts by day")
    counts = pred.groupby(["day", "decision"]).size().unstack("decision").fillna(0)
    st.bar_chart(counts)

    st.subheader("Distribution of p(up)")
    st.dataframe(pred[["ts_utc", "prob_up", "used_threshold", "decision", "st_direction", "regime_high_vol"]].tail(500),
                 use_container_width=True)

# ---------------- Drift Center ----------------
elif page == "Drift Center":
    # Latest week defaults
    weeks = load_df("SELECT DISTINCT week_start FROM drift_pred_weekly ORDER BY week_start DESC LIMIT 52")
    if weeks.empty:
        st.info("No drift tables populated yet. Run drift_weekly.py.")
        st.stop()

    week = st.sidebar.selectbox("Week start (Monday)", weeks["week_start"].tolist())

    ks = load_df(
        "SELECT ticker, ks_stat, proba_mean_shift, proba_std_shift, flag "
        "FROM drift_pred_weekly WHERE week_start=? ORDER BY ks_stat DESC",
        (week,),
    )
    psi = load_df(
        "SELECT ticker, feature, psi, flag "
        "FROM drift_feature_weekly WHERE week_start=?",
        (week,),
    )

    c1, c2, c3 = st.columns(3)
    if not ks.empty:
        c1.metric("Tickers flagged", int((ks["flag"] != "OK").sum()))
        c2.metric("Max KS", f"{ks['ks_stat'].max():.3f}")
    else:
        c1.metric("Tickers flagged", "n/a")
        c2.metric("Max KS", "n/a")

    if not psi.empty:
        c3.metric("PSI rows", len(psi))
    else:
        c3.metric("PSI rows", "n/a")

    st.subheader(f"KS Drift (week starting {week})")
    if ks.empty:
        st.info("No KS rows for this week.")
    else:
        st.dataframe(ks, use_container_width=True)

    st.subheader(f"PSI Feature Drift (week starting {week})")
    if psi.empty:
        st.info("No PSI rows for this week.")
    else:
        # Heatmap-style pivot (works best when top features are limited)
        top_n = st.sidebar.slider("Top features per ticker (for heatmap)", 5, 50, 20)
        psi_top = (
            psi.sort_values(["ticker", "psi"], ascending=[True, False])
              .groupby("ticker")
              .head(top_n)
        )
        pivot = psi_top.pivot_table(index="ticker", columns="feature", values="psi", aggfunc="max")
        st.dataframe(psi_top.sort_values(["psi"], ascending=False), use_container_width=True)
        st.markdown("**PSI pivot (ticker × feature)**")
        st.dataframe(pivot.fillna(""), use_container_width=True)

# ---------------- Model Registry ----------------
elif page == "Model Registry":
    reg = load_df("SELECT * FROM model_registry ORDER BY trained_at_utc DESC LIMIT 500")
    if reg.empty:
        st.info("No model_registry rows found. Train models first (trainer_wfo.py).")
        st.stop()

    st.subheader("Model Registry (latest 500)")
    st.dataframe(reg, use_container_width=True)

    t = st.selectbox("Ticker", list(cfg.TICKERS))
    reg_t = reg[reg["ticker"] == t].copy()
    st.subheader(f"Recent versions: {t}")
    st.dataframe(reg_t.head(50), use_container_width=True)

    if not reg_t.empty:
        row = reg_t.iloc[0]
        tf_label = normalize_timeframe_label(cfg.BAR_SIZE)
        st_row = load_df(
            "SELECT run_id, as_of, atr_len, mult FROM st_param_recommendations WHERE symbol=? AND timeframe=? ORDER BY as_of DESC LIMIT 1",
            (t, tf_label),
        )
        if not st_row.empty:
            st.metric("ST Params Version", str(st_row.iloc[0]["run_id"]))

        st.subheader("Holdout Metrics (latest)")
        try:
            st.json(json.loads(row["holdout_metrics_json"] or "{}"))
        except Exception:
            st.write(row["holdout_metrics_json"])

        st.subheader("CV Scheme (latest)")
        try:
            st.json(json.loads(row["cv_scheme_json"] or "{}"))
        except Exception:
            st.write(row["cv_scheme_json"])

# ---------------- Supertrend Params ----------------
else:
    latest = load_df(
        """
        SELECT * FROM st_param_recommendations
        WHERE (symbol, timeframe, as_of) IN (
            SELECT symbol, timeframe, MAX(as_of) FROM st_param_recommendations GROUP BY symbol, timeframe
        )
        ORDER BY symbol ASC
        """
    )

    if latest.empty:
        st.info("No supertrend recommendations saved yet. Run trainer_wfo.py to populate.")
    else:
        st.metric("Symbols covered", latest["symbol"].nunique())
        st.metric("Timeframes", latest["timeframe"].nunique())
        st.dataframe(latest, use_container_width=True)
