# ibkr-ML

A thin end-to-end workflow for training, registering, and monitoring equity direction models using Interactive Brokers (IBKR) historical bars. The pipeline fetches 2-hour bars, engineers technical features, tunes XGBoost classifiers with walk-forward evaluation, and persists both model artifacts and monitoring metrics to SQLite.

## Repository layout
- `config.py` – central hyperparameters and IB connection settings used across training and monitoring.
- `ibkr_data.py` – fetches historical bars from IBKR, caches them per ticker in SQLite, and merges optional implied volatility data.
- `features.py` – technical feature engineering (RSI, MACD, Bollinger Bands, ATR, SuperTrend, regime flags) and forward-return labeling.
- `metrics.py` – trading-oriented metrics with cost adjustments plus constraint-aware objective helper for model selection.
- `trainer_wfo.py` – end-to-end training with Optuna tuning, SHAP-based feature pruning, holdout evaluation, walk-forward summary, and model registry persistence.
- `db_schema.py` – SQLite schema for model registry, monitoring metrics, and drift indicators; helper to initialize the database.
- `monitor.py` – daily data quality checks and holdout-proxy health logging into SQLite.
- `dashboard_monitor.py` – Streamlit dashboard for visualizing monitoring metrics and registry entries.
- `REVIEW.md` – prior review notes and recommendations.

## Requirements
- Python 3.10+
- Packages: `ib_insync`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `python-dateutil`, `streamlit`, `shap` (optional but recommended for feature pruning)
- Running IB Gateway or TWS with API enabled (paper trading recommended)

Install dependencies (adjust as needed):

```bash
pip install ib_insync pandas numpy scikit-learn xgboost optuna python-dateutil streamlit shap
```

## Configuration
Adjust `TrainConfig` in `config.py` to control tickers, IB connection details, bar size, labeling parameters, tuning budget, and monitoring thresholds. Key options include:
- `TICKERS`: ticker universe to train and monitor.
- `DATA_DB`: SQLite cache/metrics path.
- `BAR_SIZE`, `WHAT_TO_SHOW`, `USE_RTH`: historical bar request parameters.
- Labeling: `LABEL_HORIZON_HOURS/BARS`, `LABEL_ATR_K`, `MIN_LABEL_PCT`.
- Objective constraints: `PROBA_SIGNAL_TH`, `MIN_SIGNALS`, `MIN_PRECISION`, `COST_BPS`.
- Tuning/CV: `N_TRIALS`, `CV_N_FOLDS`, `CV_VAL_LEN_DAYS`, `CV_GAP_DAYS`, `CV_TRAIN_MIN_BARS`, `CV_MODE`, `FAST_MODE`.
- Walk-forward: `WFO_TRAIN_MONTHS`, `WFO_VAL_MONTHS`, `WFO_STEP_MONTHS`.
- Monitoring/retrain cadence: `RETRAIN_FREQ_DAYS`, `TRAIN_LOOKBACK_MONTHS`.

## Data caching
Historical data is fetched from IBKR and cached per ticker in `DATA_DB` using `ibkr_data.fetch_and_cache`. Subsequent runs fetch only recent bars (`duration_refresh`) unless the cache is empty. Optional implied volatility bars are merged when available.

## Training workflow
1. Start IB Gateway/TWS and ensure API connectivity matches `IB_HOST`, `IB_PORT`, and `IB_CLIENT_ID` in `config.py`.
2. Run the trainer (paper trading context recommended):

   ```bash
   python trainer_wfo.py
   ```

3. For each ticker, the script will:
   - Fetch and cache bars.
   - Engineer features and forward-return labels.
   - Tune XGBoost hyperparameters with timestamp-based CV folds that include an embargo gap (Optuna) unless `FAST_MODE` is enabled.
   - Prune to the most important features via SHAP (fallback to gain-based ranking if SHAP is unavailable).
   - Train the final model, evaluate on a holdout split, and summarize walk-forward windows.
   - Save the model and metadata into `multi_models/` and record the run in the SQLite `model_registry` table.

Artifacts:
- `multi_models/<TICKER>_<BAR>_<VERSION>.json` – saved XGBoost model.
- `multi_models/<TICKER>_features.txt` – final feature list.
- `multi_models/<TICKER>_meta.json` – training metadata, metrics, and parameters.

## Monitoring
Run daily (or after new data loads) to compute data quality flags and log holdout-proxy health:

```bash
python monitor.py
```

The script calculates missing/outlier bar rates over the last 30 days and writes a row per ticker into `monitoring_metrics_daily`. Model registry lookups provide holdout metrics as a health proxy.

## Dashboard
Visualize monitoring trends and recent registry entries via Streamlit:

```bash
streamlit run dashboard_monitor.py
```

Pages:
- **Overview** – aggregate health across tickers with trend charts.
- **Ticker Drilldown** – per-ticker daily metrics and time series.
- **Model Registry** – latest training runs and JSON details for holdout/WFO summaries.

## Database initialization
`ensure_schema` in `db_schema.py` is invoked by training and monitoring scripts to create tables if they do not exist. The default SQLite file path is `market_data_cache.db` (configurable via `DATA_DB`).

## Safety notes
- Use paper trading endpoints while validating models; production trading requires additional safeguards (order throttles, position/risk controls) not included here.
- IBKR rate limits and pacing violations apply—keep bar requests reasonable and reuse caches where possible.
- Verify cost assumptions (`COST_BPS`) and objective thresholds (`MIN_SIGNALS`, `MIN_PRECISION`) before live deployment.
