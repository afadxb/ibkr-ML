# db_schema.py
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
  wfo_metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS actions_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  ticker TEXT NOT NULL,
  level TEXT NOT NULL,          -- WARN/ACTION/INFO
  action TEXT NOT NULL,         -- NONE/THROTTLE/RETRAIN/DISABLE
  reason TEXT NOT NULL,
  details_json TEXT
);

CREATE TABLE IF NOT EXISTS monitoring_metrics_daily (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,           -- YYYY-MM-DD
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
  drift_flag TEXT,              -- OK/WARN/ACTION
  action TEXT,                  -- NONE/THROTTLE/RETRAIN/DISABLE
  notes TEXT
);

CREATE TABLE IF NOT EXISTS drift_feature_weekly (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  week_start TEXT NOT NULL,     -- YYYY-MM-DD
  ticker TEXT NOT NULL,
  feature TEXT NOT NULL,
  psi REAL,
  flag TEXT                     -- OK/WARN/ACTION
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
"""

def ensure_schema(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        conn.commit()
    finally:
        conn.close()
