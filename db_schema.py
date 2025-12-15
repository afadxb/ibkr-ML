import sqlite3

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS model_registry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL,
  model_version TEXT NOT NULL,
  trained_at_utc TEXT NOT NULL,
  train_start TEXT,
  train_end TEXT,
  bar_size TEXT,
  label_horizon_hours INTEGER,
  label_params_json TEXT,
  best_params_json TEXT,
  features_checksum TEXT,
  features_count INTEGER,
  holdout_metrics_json TEXT,
  wfo_metrics_json TEXT,
  cv_scheme_json TEXT
);

CREATE TABLE IF NOT EXISTS actions_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  level TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  details_json TEXT
);

CREATE TABLE IF NOT EXISTS monitoring_metrics_daily (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  signals INTEGER,
  avg_proba REAL,
  proba_std REAL,
  precision REAL,
  expectancy_net REAL,
  profit_factor_net REAL,
  win_rate REAL,
  missing_bars_pct REAL,
  outlier_bars_pct REAL,
  drift_flag TEXT,
  action TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS drift_feature_weekly (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  week_start TEXT NOT NULL,
  ticker TEXT NOT NULL,
  feature TEXT NOT NULL,
  psi REAL,
  flag TEXT
);

CREATE TABLE IF NOT EXISTS drift_pred_weekly (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  week_start TEXT NOT NULL,
  ticker TEXT NOT NULL,
  proba_mean_shift REAL,
  proba_std_shift REAL,
  ks_stat REAL,
  flag TEXT
);

CREATE TABLE IF NOT EXISTS predictions_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  model_version TEXT NOT NULL,
  features_checksum TEXT NOT NULL,
  prob_up REAL NOT NULL,
  used_threshold REAL NOT NULL,
  decision TEXT NOT NULL,
  st_direction INTEGER,
  regime_high_vol INTEGER
);

CREATE TABLE IF NOT EXISTS st_param_recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  atr_len INTEGER NOT NULL,
  mult REAL NOT NULL,
  score REAL NOT NULL,
  median_e REAL,
  median_mdd REAL,
  median_flip_rate REAL,
  fold_count INTEGER NOT NULL,
  as_of TEXT NOT NULL,
  run_id TEXT NOT NULL,
  cost_bps REAL NOT NULL,
  notes TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS st_param_recommendations_uq ON st_param_recommendations(symbol, timeframe, run_id);
CREATE INDEX IF NOT EXISTS st_param_recommendations_latest ON st_param_recommendations(symbol, timeframe, as_of DESC);
"""

def ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()
