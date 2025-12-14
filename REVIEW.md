# Codebase Review and Recommendations

## High-level gaps
- The README only contains the project name, leaving usage, setup, and operational guidance undocumented; documenting the workflows (data fetch, training, monitoring, dashboard) would improve onboarding. 【F:README.md†L1-L1】
- There is no declared dependency list despite heavy external usage (e.g., `optuna`, `xgboost`, `ib_insync`, `streamlit`), so adding a pinned `requirements.txt` or `pyproject.toml` would make environments reproducible. 【F:trainer_wfo.py†L11-L21】【F:dashboard_monitor.py†L3-L8】

## Data acquisition and caching
- `get_cached_data` swallows all exceptions and returns an empty frame, which can hide schema drift or corruption; log the error and surface it so the caller can decide whether to continue. 【F:ibkr_data.py†L6-L15】
- `fetch_and_cache` lacks retries/backoff around `reqHistoricalData`, and it immediately raises when no bars are returned; incorporating retry logic, clearer error messages, and validation of timezone/regularity before caching would make the pipeline more resilient. 【F:ibkr_data.py†L25-L76】
- Consider normalizing timestamps to timezone-aware UTC and persisting metadata (e.g., contract parameters, `durationStr`, RTH flag) alongside cached tables to improve reproducibility of training slices. 【F:ibkr_data.py†L44-L76】

## Feature engineering and labeling
- `supertrend` iterates row-by-row, which is slow on long histories; vectorizing the logic or using NumPy operations would speed feature generation. 【F:features.py†L20-L47】
- `add_features` assumes the index is datetime-like when deriving day-of-week; guard with validation or explicit conversion to avoid failures when upstream data sources change the index type or timezone. 【F:features.py†L104-L106】
- Labeling derives thresholds from ATR but does not cap lookback gaps caused by missing bars; inserting checks that ensure minimum bar coverage per window would avoid noisy labels. 【F:features.py†L125-L135】

## Model training workflow
- Time-series CV uses `TimeSeriesSplit` without a gap, so adjacent train/validation windows can bleed information on bar-aligned features; introducing a gap parameter or rolling validation based on timestamps would better respect temporal ordering. 【F:trainer_wfo.py†L76-L99】
- Walk-forward evaluation keeps a single set of tuned hyperparameters across all windows, which may understate regime shifts; optionally re-tuning per window (or at a lower cadence) and logging those metrics would improve monitoring fidelity. 【F:trainer_wfo.py†L228-L258】
- Model metadata persisted to disk does not capture the selected feature list inside the registry row; storing the feature checksum is helpful, but also persisting the actual feature names (or a reference to the saved file) in the DB would aid reproducibility. 【F:trainer_wfo.py†L268-L316】

## Monitoring and observability
- The monitoring pipeline uses holdout metrics as a proxy for production health and applies simple thresholding; incorporating live prediction drift (e.g., probability means/KS tests) and data drift tests per feature would provide earlier warnings. 【F:monitor.py†L60-L90】
- Alerting rules only consider missing/outlier bars; extending them to cover stale models (age from registry), dropping signal counts, and portfolio-level metrics would make the dashboard more actionable. 【F:monitor.py†L46-L115】【F:config.py†L6-L54】

## Operations and safety
- Configuration is hard-coded in `TrainConfig`; supporting environment overrides (e.g., via `.env`/CLI flags) would make it easier to separate local experimentation from production runs. 【F:config.py†L6-L54】
- None of the scripts enforce deterministic seeding beyond model parameters; seeding NumPy/Pandas and setting `PYTHONHASHSEED` during tuning/runs would improve reproducibility of results. 【F:trainer_wfo.py†L11-L21】【F:trainer_wfo.py†L174-L206】
- Add unit tests for feature generation, labeling, and metric calculations so refactors (e.g., vectorizing `supertrend`) can be verified quickly. 【F:features.py†L20-L135】【F:metrics.py†L1-L37】
