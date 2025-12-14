from dataclasses import dataclass

@dataclass(frozen=True)
class BotConfig:
    # Universe
    #TICKERS: tuple = ("TSLA","NVDA","META","AAPL","AMD","GOOGL","AMZN","MSFT","PLTR","AVGO")
    TICKERS: tuple = ("TSLA","NVDA")
                  
    # Storage
    DATA_DB: str = "market_data_cache.db"
    MODEL_DIR: str = "models"

    # Data source / bars (must match training + live)
    BAR_SIZE: str = "2 hours"
    WHAT_TO_SHOW: str = "TRADES"
    USE_RTH: bool = True

    # IBKR
    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 4002
    IB_CLIENT_ID_TRAIN: int = 3
    IB_CLIENT_ID_LIVE: int = 11

    # Feature engineering (single source of truth in features.py)
    REGIME_ROLL: int = 240
    SUPERTREND_PERIOD: int = 14
    SUPERTREND_MULTIPLIER: float = 4.0

    # Labeling (training only)
    LABEL_HORIZON_HOURS: int = 12
    LABEL_HORIZON_BARS: int = 6          # 12h at 2h bars
    LABEL_ATR_K: float = 0.25
    MIN_LABEL_PCT: float = 0.001         # 0.10%

    # Model / inference thresholds
    PROBA_SIGNAL_TH: float = 0.55

    # Tuning / CV
    FAST_MODE: bool = False
    N_TRIALS: int = 50
    N_SPLITS: int = 6

    # CV gap/embargo (timestamp-based folds)
    CV_VAL_DAYS: int = 30
    CV_GAP_DAYS: int = 3
    CV_MIN_TRAIN_BARS: int = 1500
    CV_MODE: str = "expanding"           # expanding | rolling
    CV_ROLLING_TRAIN_MONTHS: int = 18

    # Constrained trading objective
    MIN_SIGNALS: int = 30
    MIN_PRECISION: float = 0.58

    # Feature pruning
    SHAP_TOP_K: int = 30

    # Holdout / WFO
    HOLDOUT_PCT: float = 0.20
    WFO_TRAIN_MONTHS: int = 18
    WFO_VAL_MONTHS: int = 1
    WFO_STEP_MONTHS: int = 1

    # Costs (net evaluation proxies)
    COST_BPS: float = 2.0

    # Live cadence / history
    RUN_EVERY_SECONDS: int = 60 * 30
    MIN_HISTORY_BARS: int = 600

    # Live policy gates (Supertrend is base indicator)
    USE_SUPERTREND_GATE: bool = True
    USE_VOLATILITY_GATE: bool = True
    HIGH_VOL_TH_BUMP: float = 0.05

    # Execution/risk (live)
    PAPER_MODE: bool = True
    ALLOW_SHORTS: bool = False
    RISK_PER_TRADE: float = 0.01
    ATR_MULT_STOP: float = 2.0
    ALLOC_PER_SYMBOL: float = 0.10

    # Notifications
    PUSHOVER_TOKEN: str = ""
    PUSHOVER_USER: str = ""

    # Drift monitoring (weekly)
    PSI_BINS: int = 10
    PSI_WARN: float = 0.10
    PSI_ACTION: float = 0.20
    KS_WARN: float = 0.10
    KS_ACTION: float = 0.20
    DRIFT_BASELINE_PRED_DAYS: int = 30
