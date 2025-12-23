from __future__ import annotations
import time
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from ib_insync import IB, Stock, Order

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

def get_open_positions(ib: IB) -> dict:
    positions = {}
    for pos in ib.positions():
        symbol = getattr(pos.contract, "symbol", None)
        qty = getattr(pos, "position", 0)
        if symbol and qty:
            positions[symbol] = qty
    return positions

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


def place_order_with_stop(ib: IB, symbol: str, market_price: float, qty: int, stop_price: float, cfg: BotConfig):
    """Place a single-leg limit buy offset above the market, with a protective stop."""
    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    contract = qualified[0] if qualified else contract

    min_tick = getattr(contract, "minTick", None)
    tick_size = min_tick if min_tick and min_tick > 0 else 0.01

    raw_limit_price = market_price + (tick_size * 2)
    limit_price = round(raw_limit_price / tick_size) * tick_size

    stop_price = max(tick_size, min(stop_price, limit_price - tick_size))
    stop_price = round(stop_price / tick_size) * tick_size

    limit_price = round(limit_price, 2)
    stop_price = round(stop_price, 2)

    if cfg.PAPER_MODE:
        paper_msg = f"[PAPER] BUY {symbol} qty={qty} LMT {limit_price:.2f} with stop {stop_price:.2f}"
        print(paper_msg)
        send_pushover("Paper order placed", paper_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)
        return

    try:
        parent_id = ib.client.getReqId()
        parent = Order()
        parent.orderId = parent_id
        parent.action = "BUY"
        parent.orderType = "LMT"
        parent.lmtPrice = limit_price
        parent.totalQuantity = qty
        parent.tif = "DAY"
        parent.transmit = False

        stop_order = Order()
        stop_order.orderId = parent_id + 1
        stop_order.action = "SELL"
        stop_order.orderType = "STP"
        stop_order.auxPrice = stop_price
        stop_order.totalQuantity = qty
        stop_order.tif = "GTC"
        stop_order.parentId = parent_id
        stop_order.transmit = True

        ib.client.placeOrder(parent.orderId, contract, parent)
        ib.client.placeOrder(stop_order.orderId, contract, stop_order)

        success_msg = (
            f"BUY {symbol} qty={qty} LMT {limit_price:.2f} placed with stop {stop_price:.2f}"
        )
        print(success_msg)
        send_pushover("Order placed", success_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)
    except Exception as exc:
        err_msg = f"Order error for {symbol}: {exc}"
        print(err_msg)
        send_pushover("Order error", err_msg, cfg.PUSHOVER_TOKEN, cfg.PUSHOVER_USER)

def main():
    cfg = BotConfig()

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
            open_positions = get_open_positions(ib)

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
                    if t in open_positions:
                        print(f"{t}: existing position size {open_positions[t]}; skip new entry.")
                        continue

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
                    if not np.isfinite(stop):
                        print(f"{t}: invalid stop; skip.")
                        continue
                    stop = max(0.01, stop)
                    place_order_with_stop(ib, t, px, qty, stop, cfg)

            time.sleep(cfg.RUN_EVERY_SECONDS)

    finally:
        ib.disconnect()
        print("Disconnected IBKR")

if __name__ == "__main__":
    main()
