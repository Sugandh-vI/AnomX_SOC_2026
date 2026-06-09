# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

ROLLING_WINDOWS = config["features"]["rolling_windows"]
Z_THRESH        = config["features"]["z_score_threshold"]


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _last_event_ts(group: pd.DataFrame, etype: str) -> pd.Series:
    """For each row, find timestamp of last occurrence of etype before this row."""
    ts_vals = group["timestamp"].values
    types   = group["event_type"].values
    result  = [pd.NaT] * len(group)
    last    = pd.NaT
    for i in range(len(group)):
        if types[i] == etype and i > 0:
            result[i] = last
        elif i > 0:
            result[i] = last
        if types[i] == etype:
            last = ts_vals[i]
    return pd.Series(result, index=group.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── 1. Time-delta features ────────────────────────────────────────────────
    logger.info("  Building time-delta features...")

    # Seconds since previous event (any type)
    df["time_since_last_event_sec"] = (
        df.groupby("user_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )

    # Seconds since last login — using simple groupby shift+ffill
    def time_since_etype(group, etype):
        mask = group["event_type"] == etype
        last_ts = group["timestamp"].where(mask).ffill().shift(1)
        delta = (group["timestamp"] - last_ts).dt.total_seconds().fillna(0).clip(lower=0)
        return delta

    df["time_since_last_login_sec"] = (
        df.groupby("user_id", group_keys=False)
          .apply(lambda g: time_since_etype(g, "login"))
    )

    df["time_since_last_deposit_sec"] = (
        df.groupby("user_id", group_keys=False)
          .apply(lambda g: time_since_etype(g, "deposit"))
    )

    # ── 2. Rolling window features ────────────────────────────────────────────
    logger.info("  Building rolling window features...")

    trade_df = df[df["event_type"] == "trade"].copy()
    sess_df  = df[df["event_type"] == "session"].copy()

    for w in ROLLING_WINDOWS:
        for col, agg in [("trade_volume", "mean"), ("trade_volume", "std"), ("pnl", "mean")]:
            col_name = f"roll_{w}_{col.replace('_volume','_vol')}_{agg}"
            rolled = (
                trade_df.groupby("user_id")[col]
                        .rolling(w, min_periods=1)
                        .agg(agg)
                        .reset_index(level=0, drop=True)
                        .rename(col_name)
            )
            trade_df = trade_df.join(rolled)

        cr_col = f"roll_{w}_click_rate_mean"
        rolled_cr = (
            sess_df.groupby("user_id")["click_rate_per_min"]
                   .rolling(w, min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
                   .rename(cr_col)
        )
        sess_df = sess_df.join(rolled_cr)

    trade_roll_cols = [c for c in trade_df.columns if c.startswith("roll_")]
    sess_roll_cols  = [c for c in sess_df.columns  if c.startswith("roll_")]

    df = df.merge(trade_df[trade_roll_cols], left_index=True, right_index=True, how="left")
    df = df.merge(sess_df[sess_roll_cols],   left_index=True, right_index=True, how="left")

    # ── 3. Burst count features ───────────────────────────────────────────────
    logger.info("  Building burst count features...")

    def burst_count(group: pd.DataFrame, window_min: int) -> pd.Series:
        ts = group["timestamp"]
        counts = []
        ts_list = ts.tolist()
        for i, t in enumerate(ts_list):
            cutoff = t - pd.Timedelta(minutes=window_min)
            counts.append(sum(1 for x in ts_list[:i+1] if x >= cutoff))
        return pd.Series(counts, index=group.index)

    df["burst_count_5min"]  = df.groupby("user_id", group_keys=False).apply(
        lambda g: burst_count(g, 5))
    df["burst_count_30min"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: burst_count(g, 30))

    # ── 4. Login anomaly features ─────────────────────────────────────────────
    logger.info("  Building login anomaly features...")

    login_df = df[df["event_type"] == "login"].copy()

    def rolling_nunique(series, w):
        return series.rolling(w, min_periods=1).apply(lambda x: len(set(x)), raw=False)

    # Encode categoricals as numeric for rolling nunique
    for col, new_col in [("ip_address", "unique_ips_last_10_logins"),
                          ("country",    "unique_countries_last_10_logins"),
                          ("device",     "unique_devices_last_10_logins")]:
        login_df[col + "_enc"] = login_df.groupby("user_id")[col].transform(
            lambda s: pd.factorize(s)[0]
        )
        rolled = (
            login_df.groupby("user_id")[col + "_enc"]
                    .rolling(10, min_periods=1)
                    .apply(lambda x: len(set(x)), raw=False)
                    .reset_index(level=0, drop=True)
                    .rename(new_col)
        )
        login_df = login_df.join(rolled)

    rolling_fails = (
        login_df.groupby("user_id")["failed_attempts"]
                .rolling(5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .rename("rolling_failed_attempts_5")
    )
    login_df = login_df.join(rolling_fails)

    login_feat_cols = ["unique_ips_last_10_logins", "unique_countries_last_10_logins",
                       "unique_devices_last_10_logins", "rolling_failed_attempts_5"]
    df = df.merge(login_df[login_feat_cols], left_index=True, right_index=True, how="left")

    # ── 5. Deposit / withdrawal ratio ─────────────────────────────────────────
    logger.info("  Building deposit/withdrawal ratio features...")

    fin_df = df[df["event_type"].isin(["deposit", "withdrawal"])].copy()

    dep_roll = (
        fin_df[fin_df["event_type"] == "deposit"]
        .groupby("user_id")["amount"]
        .rolling(5, min_periods=1).sum()
        .reset_index(level=0, drop=True)
        .rename("roll_5_deposit_sum")
    )
    fin_df = fin_df.join(dep_roll)
    fin_df["roll_5_deposit_sum"] = fin_df.groupby("user_id")["roll_5_deposit_sum"].ffill().fillna(0)

    fin_df["withdrawal_to_deposit_ratio"] = np.where(
        fin_df["event_type"] == "withdrawal",
        fin_df["amount"] / (fin_df["roll_5_deposit_sum"] + 1e-9),
        0.0
    )

    df = df.merge(
        fin_df[["roll_5_deposit_sum", "withdrawal_to_deposit_ratio"]],
        left_index=True, right_index=True, how="left"
    )

    # ── 6. Z-score features ───────────────────────────────────────────────────
    logger.info("  Building z-score features...")

    for col, name in [("trade_volume",        "trade_vol_zscore"),
                       ("pnl",                 "pnl_zscore"),
                       ("amount",              "amount_zscore"),
                       ("session_duration_mins","session_duration_zscore")]:
        df[name] = df.groupby("user_id")[col].transform(_safe_zscore)

    # ── 7. Fill NaNs ──────────────────────────────────────────────────────────
    new_cols = [c for c in df.columns if c not in set(pd.read_csv(
        config["data"]["raw_path"], nrows=0).columns)]
    df[new_cols] = df[new_cols].fillna(0)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df


def run_feature_pipeline():
    raw_path = config["data"]["raw_path"]
    out_path = config["data"]["processed_path"]

    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    df_feat = build_features(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False)

    logger.info(f"Saved features to {out_path}")

    orig_cols = set(pd.read_csv(raw_path, nrows=0).columns)
    new_cols  = [c for c in df_feat.columns if c not in orig_cols]
    print(f"\n✅ Features saved to {out_path}")
    print(f"   Original columns : {len(orig_cols)}")
    print(f"   New columns added: {len(new_cols)}")
    print(f"   Total columns    : {len(df_feat.columns)}")
    print(f"\nNew features:")
    for c in new_cols:
        print(f"  + {c}")
    return df_feat


if __name__ == "__main__":
    run_feature_pipeline()