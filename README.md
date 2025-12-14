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
