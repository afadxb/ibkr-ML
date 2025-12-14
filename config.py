# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    # Universe
    TICKERS: tuple = ("TSLA","NVDA","META","AAPL","AMD","GOOGL","AMZN","MSFT","PLTR","AVGO")

    # Data
    DATA_DB: str = "market_data_cache.db"
    MODEL_DIR: str = "multi_models"
    BAR_SIZE: str = "2 hours"
    WHAT_TO_SHOW: str = "TRADES"
    USE_RTH: bool = True

    # IBKR
    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 4002
    IB_CLIENT_ID: int = 3

    # Labeling (2h bars)
    LABEL_HORIZON_HOURS: int = 12         # 12h forecast
    LABEL_HORIZON_BARS: int = 6           # 12h / 2h
    LABEL_ATR_K: float = 0.25
    MIN_LABEL_PCT: float = 0.001          # 0.10%

    # Regime
    REGIME_ROLL: int = 240                # ~20 trading days on 2h bars (12 bars/day)

    # Tuning / CV
    FAST_MODE: bool = False
    N_TRIALS: int = 50
    N_SPLITS: int = 6

    # Trading objective constraints
    PROBA_SIGNAL_TH: float = 0.55
    MIN_SIGNALS: int = 30
    MIN_PRECISION: float = 0.58

    # Feature pruning
    SHAP_TOP_K: int = 30

    # WFO / Holdout
    HOLDOUT_PCT: float = 0.20
    WFO_TRAIN_MONTHS: int = 18
    WFO_VAL_MONTHS: int = 1
    WFO_STEP_MONTHS: int = 1

    # Scheduled retraining
    RETRAIN_FREQ_DAYS: int = 7
    TRAIN_LOOKBACK_MONTHS: int = 18

    # Costs for net evaluation (conservative defaults)
    COST_BPS: float = 2.0                 # 0.02% per round-trip proxy; adjust later
