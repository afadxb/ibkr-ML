import numpy as np
from sklearn.metrics import precision_score, roc_auc_score

def apply_costs(fwd_ret: np.ndarray, cost_bps: float) -> np.ndarray:
    return fwd_ret - (cost_bps / 10000.0)

def trading_metrics(y_true, proba, fwd_ret, proba_th: float, cost_bps: float):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    fwd_ret = np.asarray(fwd_ret)

    sig = proba >= proba_th
    signals = int(sig.sum())

    try:
        auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    y_pred = (proba >= 0.5).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    if signals == 0:
        return {
            "signals": 0,
            "precision": precision,
            "roc_auc": auc,
            "expectancy_net": float("nan"),
            "profit_factor_net": float("nan"),
            "win_rate": float("nan"),
            "avg_fwd_ret_on_signals_net": float("nan"),
        }

    ret_sig = apply_costs(fwd_ret[sig], cost_bps)
    wins = ret_sig[ret_sig > 0].sum()
    losses = ret_sig[ret_sig < 0].sum()
    pf = float(wins / abs(losses)) if losses < 0 else float("inf")
    win_rate = float((ret_sig > 0).mean())
    expectancy = float(np.nanmean(ret_sig))

    return {
        "signals": signals,
        "precision": precision,
        "roc_auc": auc,
        "expectancy_net": expectancy,
        "profit_factor_net": pf,
        "win_rate": win_rate,
        "avg_fwd_ret_on_signals_net": float(np.nanmean(ret_sig)),
    }

def constrained_fold_score(m: dict, min_signals: int, min_precision: float) -> float:
    if m["signals"] < min_signals:
        return -1.0 - (min_signals - m["signals"]) * 0.01
    if m["precision"] < min_precision:
        return -0.5 - (min_precision - m["precision"]) * 1.0
    return float(m["avg_fwd_ret_on_signals_net"])
