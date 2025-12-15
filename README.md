# SteadyAlpha ML Sync (Training + Live)

This refactor enforces full synchronization:
- Single source of truth feature pipeline: features.py
- Single config: config.py (BotConfig)
- Shared modules imported by both training and live
- Strict feature schema + checksum enforcement
- Predictions logged for real drift monitoring (KS)
- Weekly drift job populates PSI + KS tables

## Run order
1) Train models + registry:
   python trainer_wfo.py
2) Run live bot (paper by default):
   python live_bot.py
3) Weekly drift (PSI + KS):
   python drift_weekly.py

## Supertrend parameter recommendations
- The walk-forward selector inside `trainer_wfo.py` runs a grid over 2h bars and writes the best `atr_len`/`mult` per symbol to the shared SQLite table `st_param_recommendations`.
- A JSON backup artifact is also dropped under `models/st_params_<timeframe>_<run_id>.json` for transparency, but SQLite remains the source of truth.
- `features.apply_supertrend_with_reco` loads the latest params (or falls back to config defaults) so both training and live inference share one Supertrend calculation.
- `live_bot.py` caches the loaded params at startup per symbol/timeframe; no intraday re-optimization.
