# src/models/isolation_forest.py

import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Feature groups (only numeric, no leakage columns) ────────────────────────

# These are the columns we actually feed into the model.
# We deliberately EXCLUDE: event_id, user_id, timestamp, is_anomalous,
# anomaly_type, ip_address, country, device, instrument, method,
# kyc_change_type (all non-numeric or label columns)

FEATURE_COLS = [
    # Raw behavioral signals
    "hour_of_day", "day_of_week", "is_weekend",
    "login_success", "failed_attempts", "timezone_gap_hours",
    "lot_size", "trade_volume", "pnl", "margin_used", "trade_duration_seconds",
    "trade_volume_vs_baseline", "is_night_trade",
    "amount", "is_immediate_withdrawal",
    "session_duration_mins", "page_clicks", "click_rate_per_min",
    "account_age_days",
    # Engineered features
    "time_since_last_event_sec", "time_since_last_login_sec",
    "time_since_last_deposit_sec",
    "roll_5_trade_vol_mean",  "roll_5_trade_vol_std",  "roll_5_pnl_mean",
    "roll_10_trade_vol_mean", "roll_10_trade_vol_std", "roll_10_pnl_mean",
    "roll_30_trade_vol_mean", "roll_30_trade_vol_std", "roll_30_pnl_mean",
    "roll_5_click_rate_mean", "roll_10_click_rate_mean", "roll_30_click_rate_mean",
    "burst_count_5min", "burst_count_30min",
    "unique_ips_last_10_logins", "unique_countries_last_10_logins",
    "unique_devices_last_10_logins", "rolling_failed_attempts_5",
    "roll_5_deposit_sum", "withdrawal_to_deposit_ratio",
    "trade_vol_zscore", "pnl_zscore", "amount_zscore", "session_duration_zscore",
]


def load_features(path: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load processed features, return (full_df, X, y_true)."""
    df = pd.read_csv(path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    X      = df[FEATURE_COLS].values.astype(np.float32)
    y_true = df["is_anomalous"].values
    return df, X, y_true


def build_pipeline() -> Pipeline:
    cfg = config["model"]["isolation_forest"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators  = cfg["n_estimators"],
            contamination = cfg["contamination"],
            random_state  = cfg["random_state"],
            n_jobs        = -1,
        ))
    ])


def train(df: pd.DataFrame, X: np.ndarray) -> tuple[Pipeline, np.ndarray]:
    logger.info(f"Training Isolation Forest on {X.shape[0]} events, {X.shape[1]} features...")
    pipe = build_pipeline()
    pipe.fit(X)

    # sklearn returns -1 (anomaly) / +1 (normal) — convert to binary
    preds  = pipe.predict(X)          # -1 or +1
    labels = (preds == -1).astype(int)  # 1 = anomaly

    # Raw decision scores (lower = more anomalous, we negate for intuition)
    raw_scores   = pipe.decision_function(X)   # negative = more anomalous
    anomaly_scores = -raw_scores               # higher = more anomalous

    logger.info(f"Flagged {labels.sum()} events as anomalous "
                f"({labels.mean()*100:.1f}% of dataset)")
    return pipe, labels, anomaly_scores


def evaluate(y_true: np.ndarray, labels: np.ndarray, anomaly_scores: np.ndarray):
    """Print a clean evaluation report."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, average_precision_score)

    print("\n" + "="*55)
    print("  ISOLATION FOREST — EVALUATION REPORT")
    print("="*55)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-"*42)
    print(f"{'Total events':<30} {len(y_true):>10,}")
    print(f"{'True anomalies':<30} {y_true.sum():>10,}")
    print(f"{'Predicted anomalies':<30} {labels.sum():>10,}")

    try:
        auc   = roc_auc_score(y_true, anomaly_scores)
        ap    = average_precision_score(y_true, anomaly_scores)
        print(f"{'ROC-AUC':<30} {auc:>10.4f}")
        print(f"{'Average Precision':<30} {ap:>10.4f}")
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")

    print("\nClassification Report:")
    print(classification_report(y_true, labels,
                                  target_names=["Normal", "Anomaly"],
                                  zero_division=0))

    cm = confusion_matrix(y_true, labels)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred Normal  Pred Anomaly")
    print(f"  Actual Normal   {cm[0,0]:>8,}      {cm[0,1]:>8,}")
    print(f"  Actual Anomaly  {cm[1,0]:>8,}      {cm[1,1]:>8,}")
    print("="*55)


def get_top_features(X_row: np.ndarray, score: float, n: int = 5) -> list[dict]:
    """
    Return top N features contributing to anomaly score for a single event.
    Simple approach: features with highest absolute z-score from the training mean.
    (Used later by the API for explainability.)
    """
    abs_vals = np.abs(X_row)
    top_idx  = np.argsort(abs_vals)[::-1][:n]
    return [
        {"feature": FEATURE_COLS[i], "value": float(X_row[i])}
        for i in top_idx
    ]


def save_model(pipe: Pipeline, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    out = Path(path) / "isolation_forest.pkl"
    joblib.dump(pipe, out)
    logger.info(f"Model saved to {out}")


def load_model(path: str) -> Pipeline:
    model_path = Path(path) / "isolation_forest.pkl"
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def run_isolation_forest():
    processed_path = config["data"]["processed_path"]
    model_path     = config["api"]["model_path"]

    df, X, y_true = load_features(processed_path)
    pipe, labels, anomaly_scores = train(df, X)
    evaluate(y_true, labels, anomaly_scores)
    save_model(pipe, model_path)

    # Save predictions back to CSV for inspection
    df["if_score"]      = anomaly_scores
    df["if_prediction"] = labels
    out = processed_path.replace("features.csv", "features_with_scores.csv")
    df.to_csv(out, index=False)
    logger.info(f"Scores saved to {out}")

    return pipe, df


if __name__ == "__main__":
    run_isolation_forest()