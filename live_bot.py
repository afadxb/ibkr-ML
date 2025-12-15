from __future__ import annotations
import time
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from ib_insync import IB, Stock
from ibapi.contract import Contract, ComboLeg
from ibapi.order import Order
from ibapi.tag_value import TagValue

from config import BotConfig
from db_schema import ensure_schema
from datahub import refresh_cache
from features import get_st_params
from pipeline import make_features
from modelio import load_bundle, build_inference_row
from policy import decide
from execution import initial_stop_from_atr, calc_position_size
from utils import utc_now_iso, send_pushover, is_regular_trading_time
from supertrend_params import normalize_timeframe_label

def get_available_funds(ib: IB) -> float:
    acct = ib.accountSummary()
    vals = [v.value for v in acct if v.tag == "AvailableFunds"]
    return float(vals[0]) if vals else 0.0

def get_last_price(ib: IB, ticker: str) -> float:
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    t = ib.reqMktData(contract, "", False, False)
    ib.sleep(1.5)
    p = float(t.marketPrice())
    return p if p and p > 0 else float("nan")

def write_prediction(cfg: BotConfig, ticker: str, model_version: str, feats_chk: str,
                     prob_up: float, used_th: float, decision_str: str,
                     st_direction: int | None, regime_high_vol: int | None):
    conn = sqlite3.connect(cfg.DATA_DB)
    try:
        conn.execute(
            "INSERT INTO predictions_log "
            "(ts_utc,ticker,model_version,features_checksum,prob_up,used_threshold,decision,st_direction,regime_high_vol) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (utc_now_iso(), ticker, model_version, feats_chk, float(prob_up), float(used_th), decision_str,
             st_direction, regime_high_vol)
        )
        conn.commit()
    finally:
        conn.close()


def build_combo_contract(long_leg: Stock, short_leg: Stock) -> Contract:
    """Create a SMART-routed combo contract for a stock pair.

    This mirrors IBKR's combo order example by creating a BAG contract with two
    stock legs and enabling NonGuaranteed routing.
    """
    combo = Contract()
    combo.symbol = f"{long_leg.symbol},{short_leg.symbol}"
    combo.secType = "BAG"
    combo.exchange = "SMART"
    combo.currency = "USD"

    leg_buy = ComboLeg()
    leg_buy.conId = long_leg.conId
    leg_buy.ratio = 1
    leg_buy.action = "BUY"
    leg_buy.exchange = "SMART"

    leg_sell = ComboLeg()
    leg_sell.conId = short_leg.conId
    leg_sell.ratio = 1
    leg_sell.action = "SELL"
    leg_sell.exchange = "SMART"

    combo.comboLegs = [leg_buy, leg_sell]
    return combo


def build_combo_order(order_id: int, qty: int, limit_price: float) -> Order:
    order = Order()
    order.orderId = order_id
    order.action = "BUY"
    order.orderType = "LMT"
    order.lmtPrice = limit_price
    order.totalQuantity = qty
    order.tif = "GTC"
    order.smartComboRoutingParams = [TagValue("NonGuaranteed", "1")]
    return order


def place_combo_order(ib: IB, long_symbol: str, short_symbol: str, qty: int, limit_price: float, cfg: BotConfig):
    """Place a combo order using IBKR's BAG contract pattern.

    This follows the IBKR "Combo Order Snippet" example by:
    - qualifying both stock legs to fetch their conIds
    - constructing a BAG contract with two ComboLeg entries
    - creating a manual limit Order with NonGuaranteed routing
    - submitting the order via the underlying EClient connection
    """
    long_leg = Stock(long_symbol, "SMART", "USD")
    short_leg = Stock(short_symbol, "SMART", "USD")
    ib.qualifyContracts(long_leg, short_leg)

    if cfg.PAPER_MODE:
        paper_msg = (
            f"[PAPER] COMBO BUY {long_symbol} / SELL {short_symbol} "
            f"qty={qty} limit_spread={limit_price:.2f}"
        )
        print(paper_msg)
        send_pushover("Paper order placed", paper_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)
        return

    try:
        combo_contract = build_combo_contract(long_leg, short_leg)
        order_id = ib.client.getReqId()
        order = build_combo_order(order_id, qty, limit_price)
        ib.client.placeOrder(order_id, combo_contract, order)

        success_msg = (
            f"COMBO BUY {long_symbol} / SELL {short_symbol} "
            f"qty={qty} limit_spread={limit_price:.2f} placed"
        )
        print(success_msg)
        send_pushover("Order placed", success_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)
    except Exception as exc:
        err_msg = f"Combo order error for {long_symbol}/{short_symbol}: {exc}"
        print(err_msg)
        send_pushover("Order error", err_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)

def main():
    cfg = BotConfig()

    if not cfg.HEDGE_TICKER:
        raise RuntimeError(
            "HEDGE_TICKER must be configured for combo execution; single-leg orders are disabled."
        )

    ensure_schema(cfg.DATA_DB)

    timeframe = normalize_timeframe_label(cfg.BAR_SIZE)
    st_cache = {}
    for t in cfg.TICKERS:
        atr_len, mult, meta = get_st_params(
            t,
            timeframe,
            cfg.DATA_DB,
            fallback=(cfg.SUPERTREND_PERIOD, cfg.SUPERTREND_MULTIPLIER),
        )
        st_cache[t] = (atr_len, mult, meta)
        if meta:
            print(
                f"ST params {t} {timeframe}: atr={atr_len} mult={mult} "
                f"score={meta.get('score'):.4f} as_of={meta.get('as_of')} run_id={meta.get('run_id')}"
            )
        else:
            print(f"ST params {t} {timeframe}: atr={atr_len} mult={mult} (fallback)")

    bundles = {t: load_bundle(t, cfg) for t in cfg.TICKERS}
    for t,b in bundles.items():
        print(f"Loaded {t}: version={b['model_version']} model={b['model_path']} feats={len(b['features'])}")

    ib = IB()
    ib.connect(cfg.IB_HOST, cfg.IB_PORT, clientId=cfg.IB_CLIENT_ID_LIVE)
    print("Connected IBKR")

    try:
        while True:
            balance = get_available_funds(ib)

            for t in cfg.TICKERS:
                b = bundles[t]

                bars = refresh_cache(ib, cfg.DATA_DB, t, cfg.BAR_SIZE, cfg.WHAT_TO_SHOW, cfg.USE_RTH)
                if len(bars) < cfg.MIN_HISTORY_BARS:
                    print(f"{t}: insufficient history ({len(bars)}<{cfg.MIN_HISTORY_BARS}). Skipping.")
                    continue

                df_feat = make_features(
                    bars,
                    cfg,
                    symbol=t,
                    timeframe=timeframe,
                    st_params=st_cache.get(t),
                ).dropna()
                if df_feat.empty:
                    print(f"{t}: features empty after dropna.")
                    continue

                X = build_inference_row(df_feat, b["features"])
                prob_up = float(b["booster"].predict(xgb.DMatrix(X))[0])

                d = decide(prob_up, df_feat, cfg)
                write_prediction(cfg, t, b["model_version"], b["features_checksum"],
                                 prob_up, d["used_threshold"], d["decision"], d["st_direction"], d["regime_high_vol"])

                print(f"{t}: p={prob_up:.3f} th={d['used_threshold']:.2f} gate={d['gate_pass']} -> {d['decision']}")

                if d["decision"] == "BUY":
                    if not is_regular_trading_time():
                        print(f"{t}: signal outside RTH; skip execution.")
                        continue

                    px = get_last_price(ib, t)
                    if not np.isfinite(px):
                        print(f"{t}: invalid market price; skip.")
                        continue
                    stop = initial_stop_from_atr(px, df_feat, cfg)
                    qty = calc_position_size(balance, px, stop, cfg)
                    if qty <= 0:
                        print(f"{t}: qty=0; skip.")
                        continue

                    hedge_px = get_last_price(ib, cfg.HEDGE_TICKER)
                    if not np.isfinite(hedge_px):
                        print(f"{cfg.HEDGE_TICKER}: invalid hedge price; skip combo order.")
                        continue
                    combo_limit = px - hedge_px
                    place_combo_order(ib, t, cfg.HEDGE_TICKER, qty, combo_limit, cfg)

            time.sleep(cfg.RUN_EVERY_SECONDS)

    finally:
        ib.disconnect()
        print("Disconnected IBKR")

if __name__ == "__main__":
    main()
