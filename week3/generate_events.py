# src/data_gen/generate_events.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yaml
from pathlib import Path

# ── Load config ───────────────────────────────────────────────────────────────
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED         = config["data"]["random_seed"]
N_USERS      = config["data"]["n_users"]
N_EVENTS     = config["data"]["n_events"]
ANOMALY_FRAC = config["data"]["anomaly_fraction"]
START_DATE   = datetime(2024, 1, 1)
END_DATE     = datetime(2024, 3, 31)

np.random.seed(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "USDCHF", "AUDUSD"]
COUNTRIES   = ["IN", "US", "UK", "SG", "AE", "NG", "RU", "CN", "DE", "BR"]
DEVICES     = ["chrome_win", "safari_mac", "android_app", "ios_app", "firefox_linux"]
EVENT_TYPES = ["login", "trade", "deposit", "withdrawal", "session", "kyc_change"]
EVENT_WEIGHTS = [0.22, 0.35, 0.15, 0.10, 0.12, 0.06]

# Timezone offsets per country (UTC offset in hours) — used for impossibility detection
COUNTRY_TIMEZONES = {
    "IN": 5.5, "US": -5, "UK": 0, "SG": 8,
    "AE": 4,   "NG": 1,  "RU": 3, "CN": 8,
    "DE": 1,   "BR": -3
}

ANOMALY_TYPES = [
    "ip_hopper",
    "wash_trader",
    "deposit_withdrawal_cycler",
    "bot_trader",
    "structurer",
    "brute_forcer",
    "dormant_withdrawer",
    "consistent_winner",
    "device_switcher",
    "kyc_manipulator",
]

# Hub IPs shared across multiple anomalous users — for network graph anomaly detection
SHARED_IPS = [f"172.16.{i}.1" for i in range(10)]

def random_timestamp(start=START_DATE, end=END_DATE):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def random_ip():
    return f"192.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def timezone_offset(country):
    return COUNTRY_TIMEZONES.get(country, 0)

# ── Build user profiles ───────────────────────────────────────────────────────
def build_user_profiles():
    profiles = {}
    n_anomalous = int(N_USERS * ANOMALY_FRAC)

    for i in range(N_USERS):
        user_id      = f"USER_{i:04d}"
        is_anomalous = i < n_anomalous
        anomaly_type = random.choice(ANOMALY_TYPES) if is_anomalous else None

        if is_anomalous and random.random() < 0.3:
            home_ip = random.choice(SHARED_IPS)
        else:
            home_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.1"

        home_country = random.choice(COUNTRIES)

        profiles[user_id] = {
            "home_ip"              : home_ip,
            "home_country"        : home_country,
            "home_timezone_offset": timezone_offset(home_country),
            "preferred_device"    : random.choice(DEVICES),
            "instruments"         : random.sample(INSTRUMENTS, k=random.randint(1, 3)),
            "typical_trade_vol"   : round(np.random.uniform(1000, 30000), 2),
            "typical_deposit"     : round(np.random.uniform(100, 3000), 2),
            "is_anomalous"        : is_anomalous,
            "anomaly_type"        : anomaly_type,
            "account_created_at"  : random_timestamp(
                START_DATE - timedelta(days=365),
                START_DATE
            ),
            "last_active"         : random_timestamp(START_DATE, START_DATE + timedelta(days=30)),
        }

    return profiles

# ── Empty event base ──────────────────────────────────────────────────────────
def empty_base(user_id, ts):
    return {
        "event_id"                    : f"EVT_{random.randint(100000,999999)}",
        "user_id"                     : user_id,
        "event_type"                  : None,
        "timestamp"                   : ts,
        "hour_of_day"                 : ts.hour,
        "day_of_week"                 : ts.weekday(),
        "is_weekend"                  : int(ts.weekday() >= 5),
        "is_anomalous"                : 0,
        "anomaly_type"                : "none",
        # Login fields
        "ip_address"                  : None,
        "country"                     : None,
        "device"                      : None,
        "login_success"               : None,
        "failed_attempts"             : None,
        # ── EXTRA: timezone gap between home country and login country
        "timezone_gap_hours"          : None,
        # Trade fields
        "instrument"                  : None,
        "lot_size"                    : None,
        "trade_volume"                : None,
        "pnl"                         : None,
        "margin_used"                 : None,
        "trade_duration_seconds"      : None,
        # ── EXTRA: ratio of this trade volume vs user's typical
        "trade_volume_vs_baseline"    : None,
        # ── EXTRA: is this a night trade (12am-5am local time)?
        "is_night_trade"              : None,
        # Deposit/withdrawal fields
        "amount"                      : None,
        "method"                      : None,
        # ── EXTRA: withdrawal to deposit ratio signal
        "is_immediate_withdrawal"     : None,
        # Session fields
        "session_duration_mins"       : None,
        "page_clicks"                 : None,
        # ── EXTRA: clicks per minute — bot detection
        "click_rate_per_min"          : None,
        # KYC fields
        "kyc_change_type"             : None,
        # ── EXTRA: account age in days at time of event
        "account_age_days"            : None,
    }

# ── Normal event generator ────────────────────────────────────────────────────
def generate_normal_event(user_id, profile, ts=None):
    if ts is None:
        ts = random_timestamp()

    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
    base = empty_base(user_id, ts)
    base["event_type"]     = event_type
    base["account_age_days"] = (ts - profile["account_created_at"]).days

    if event_type == "login":
        country = profile["home_country"]
        tz_gap  = abs(timezone_offset(country) - profile["home_timezone_offset"])
        base.update({
            "ip_address"        : profile["home_ip"],
            "country"           : country,
            "device"            : profile["preferred_device"],
            "login_success"     : 1,
            "failed_attempts"   : int(np.random.choice([0,1], p=[0.95, 0.05])),
            "timezone_gap_hours": tz_gap,
        })

    elif event_type == "trade":
        vol = profile["typical_trade_vol"]
        actual_vol = round(np.random.normal(vol, vol * 0.2), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        base.update({
            "instrument"                : random.choice(profile["instruments"]),
            "lot_size"                  : round(np.random.uniform(0.01, 2.0), 2),
            "trade_volume"              : actual_vol,
            "pnl"                       : round(np.random.normal(0, 200), 2),
            "margin_used"               : round(np.random.uniform(100, 5000), 2),
            "trade_duration_seconds"    : int(np.random.uniform(60, 3600)),
            "trade_volume_vs_baseline"  : round(actual_vol / vol, 3),
            "is_night_trade"            : int(0 <= local_hour <= 5),
        })

    elif event_type == "deposit":
        base.update({
            "amount"              : round(np.random.normal(profile["typical_deposit"], 200), 2),
            "method"              : random.choice(["card", "bank", "crypto"]),
            "is_immediate_withdrawal": 0,
        })

    elif event_type == "withdrawal":
        base.update({
            "amount"              : round(np.random.uniform(50, profile["typical_deposit"] * 0.8), 2),
            "method"              : random.choice(["bank", "crypto"]),
            "is_immediate_withdrawal": 0,
        })

    elif event_type == "session":
        duration = round(np.random.uniform(5, 60), 1)
        clicks   = int(np.random.uniform(5, 80))
        base.update({
            "session_duration_mins": duration,
            "page_clicks"          : clicks,
            "device"               : profile["preferred_device"],
            "click_rate_per_min"   : round(clicks / max(duration, 0.1), 2),
        })

    elif event_type == "kyc_change":
        base.update({
            "kyc_change_type": random.choice(["address_update", "phone_update"]),
        })

    return base

# ── Anomalous event generators ────────────────────────────────────────────────

def events_ip_hopper(user_id, profile, n=8):
    """8.1 Rapid IP + country switching — timezone_gap_hours will be high"""
    events = []
    base_ts = random_timestamp()
    prev_country = profile["home_country"]
    for i in range(n):
        ts      = base_ts + timedelta(minutes=random.randint(1, 15))
        country = random.choice(COUNTRIES)
        tz_gap  = abs(timezone_offset(country) - timezone_offset(prev_country))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"        : "login",
            "ip_address"        : random_ip(),
            "country"           : country,
            "device"            : random.choice(DEVICES),
            "login_success"     : 1,
            "failed_attempts"   : 0,
            "timezone_gap_hours": tz_gap,   # ← key signal: large gap = impossible travel
            "is_anomalous"      : 1,
            "anomaly_type"      : "ip_hopper",
        })
        prev_country = country
        events.append(e)
    return events

def events_wash_trader(user_id, profile, n=10):
    """8.3 Volume spike 10-20x baseline, single instrument, always profits"""
    events = []
    base_ts = random_timestamp()
    vol = profile["typical_trade_vol"]
    for i in range(n):
        ts         = base_ts + timedelta(minutes=i * 2)
        actual_vol = round(vol * random.uniform(10, 20), 2)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"               : "trade",
            "instrument"               : INSTRUMENTS[0],
            "lot_size"                 : round(np.random.uniform(20, 50), 2),
            "trade_volume"             : actual_vol,
            "pnl"                      : round(np.random.uniform(500, 5000), 2),
            "margin_used"              : round(np.random.uniform(50000, 200000), 2),
            "trade_duration_seconds"   : int(np.random.uniform(1, 30)),
            "trade_volume_vs_baseline" : round(actual_vol / vol, 3),  # ← will be 10-20x
            "is_night_trade"           : 0,
            "is_anomalous"             : 1,
            "anomaly_type"             : "wash_trader",
        })
        events.append(e)
    return events

def events_deposit_withdrawal_cycler(user_id, profile, n=4):
    """8.2 Deposit → immediate withdrawal, is_immediate_withdrawal=1"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts_dep = base_ts + timedelta(hours=i * 12)
        dep_amount = round(np.random.uniform(5000, 20000), 2)

        e_dep = generate_normal_event(user_id, profile, ts_dep)
        e_dep.update({
            "event_type"             : "deposit",
            "amount"                 : dep_amount,
            "method"                 : "crypto",
            "is_immediate_withdrawal": 1,
            "is_anomalous"           : 1,
            "anomaly_type"           : "deposit_withdrawal_cycler",
        })
        events.append(e_dep)

        ts_wit = ts_dep + timedelta(hours=random.randint(1, 3))
        e_wit = generate_normal_event(user_id, profile, ts_wit)
        e_wit.update({
            "event_type"             : "withdrawal",
            "amount"                 : round(dep_amount * random.uniform(0.85, 0.98), 2),
            "method"                 : "crypto",
            "is_immediate_withdrawal": 1,
            "is_anomalous"           : 1,
            "anomaly_type"           : "deposit_withdrawal_cycler",
        })
        events.append(e_wit)
    return events

def events_bot_trader(user_id, profile, n=8):
    """8.4 Inhuman click rate, very short sessions at 2-4 AM"""
    events = []
    base_ts = random_timestamp()
    base_ts = base_ts.replace(hour=random.randint(2, 4))
    for i in range(n):
        ts       = base_ts + timedelta(minutes=i * 3)
        duration = round(np.random.uniform(0.2, 1.5), 1)
        clicks   = int(np.random.uniform(300, 600))
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"           : "session",
            "session_duration_mins": duration,
            "page_clicks"          : clicks,
            "click_rate_per_min"   : round(clicks / max(duration, 0.1), 2),  # ← 200-3000/min
            "device"               : "chrome_win",
            "hour_of_day"          : ts.hour,
            "is_anomalous"         : 1,
            "anomaly_type"         : "bot_trader",
        })
        events.append(e)
    return events

def events_structurer(user_id, profile, n=12):
    """8.2 Many deposits just under 1000 — classic structuring pattern"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i * 2)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"  : "deposit",
            "amount"      : round(np.random.uniform(490, 999), 2),
            "method"      : random.choice(["card", "crypto"]),
            "is_anomalous": 1,
            "anomaly_type": "structurer",
        })
        events.append(e)
    return events

def events_brute_forcer(user_id, profile, n=6):
    """8.7 Escalating failed_attempts then sudden login success"""
    events = []
    base_ts = random_timestamp()
    for i in range(n - 1):
        ts = base_ts + timedelta(seconds=i * 30)
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"       : "login",
            "ip_address"       : random_ip(),
            "country"          : random.choice(COUNTRIES),
            "login_success"    : 0,
            "failed_attempts"  : i + 1,
            "is_anomalous"     : 1,
            "anomaly_type"     : "brute_forcer",
        })
        events.append(e)

    ts_ok = base_ts + timedelta(seconds=n * 30)
    e_ok  = generate_normal_event(user_id, profile, ts_ok)
    e_ok.update({
        "event_type"      : "login",
        "ip_address"      : random_ip(),
        "login_success"   : 1,
        "failed_attempts" : n - 1,
        "is_anomalous"    : 1,
        "anomaly_type"    : "brute_forcer",
    })
    events.append(e_ok)
    return events

def events_dormant_withdrawer(user_id, profile, n=3):
    """8.2 Long dormancy then sudden large withdrawal — account_age_days will be high"""
    events = []
    dormant_end = END_DATE - timedelta(days=random.randint(1, 5))
    for i in range(n):
        ts = dormant_end + timedelta(hours=i)
        e  = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"  : "withdrawal",
            "amount"      : round(np.random.uniform(15000, 50000), 2),
            "method"      : "crypto",
            "is_anomalous": 1,
            "anomaly_type": "dormant_withdrawer",
        })
        events.append(e)
    return events

def events_consistent_winner(user_id, profile, n=10):
    """8.3 Always positive PnL, sub-10s trades — latency arbitrage signal"""
    events = []
    base_ts = random_timestamp()
    vol = profile["typical_trade_vol"]
    for i in range(n):
        ts         = base_ts + timedelta(minutes=i * 5)
        actual_vol = round(np.random.uniform(100000, 500000), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        e = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"               : "trade",
            "instrument"               : random.choice(INSTRUMENTS[:2]),
            "lot_size"                 : round(np.random.uniform(5, 15), 2),
            "trade_volume"             : actual_vol,
            "pnl"                      : round(np.random.uniform(200, 2000), 2),
            "trade_duration_seconds"   : int(np.random.uniform(1, 10)),
            "margin_used"              : round(np.random.uniform(10000, 50000), 2),
            "trade_volume_vs_baseline" : round(actual_vol / vol, 3),
            "is_night_trade"           : int(0 <= local_hour <= 5),
            "is_anomalous"             : 1,
            "anomaly_type"             : "consistent_winner",
        })
        events.append(e)
    return events

def events_device_switcher(user_id, profile, n=8):
    """8.4 Different device fingerprint every login"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i)
        e  = generate_normal_event(user_id, profile, ts)
        e.update({
            "event_type"      : "login",
            "device"          : random.choice(DEVICES),
            "ip_address"      : profile["home_ip"],
            "country"         : profile["home_country"],
            "login_success"   : 1,
            "failed_attempts" : 0,
            "is_anomalous"    : 1,
            "anomaly_type"    : "device_switcher",
        })
        events.append(e)
    return events

def events_kyc_manipulator(user_id, profile, n=4):
    """8.7 KYC change immediately followed by large withdrawal"""
    events = []
    base_ts = random_timestamp()
    for i in range(n):
        ts = base_ts + timedelta(hours=i * 2)

        e_kyc = generate_normal_event(user_id, profile, ts)
        e_kyc.update({
            "event_type"     : "kyc_change",
            "kyc_change_type": random.choice(["bank_account_update", "address_update", "id_resubmit"]),
            "is_anomalous"   : 1,
            "anomaly_type"   : "kyc_manipulator",
        })
        events.append(e_kyc)

        ts_wit = ts + timedelta(minutes=random.randint(10, 60))  # ← small gap = suspicious
        e_wit  = generate_normal_event(user_id, profile, ts_wit)
        e_wit.update({
            "event_type"  : "withdrawal",
            "amount"      : round(np.random.uniform(10000, 40000), 2),
            "method"      : "crypto",
            "is_anomalous": 1,
            "anomaly_type": "kyc_manipulator",
        })
        events.append(e_wit)
    return events

# ── Dispatcher ────────────────────────────────────────────────────────────────
ANOMALY_GENERATORS = {
    "ip_hopper"                 : events_ip_hopper,
    "wash_trader"               : events_wash_trader,
    "deposit_withdrawal_cycler" : events_deposit_withdrawal_cycler,
    "bot_trader"                : events_bot_trader,
    "structurer"                : events_structurer,
    "brute_forcer"              : events_brute_forcer,
    "dormant_withdrawer"        : events_dormant_withdrawer,
    "consistent_winner"         : events_consistent_winner,
    "device_switcher"           : events_device_switcher,
    "kyc_manipulator"           : events_kyc_manipulator,
}

# ── Main ──────────────────────────────────────────────────────────────────────
def generate_dataset():
    print("Building user profiles...")
    profiles      = build_user_profiles()
    user_ids      = list(profiles.keys())
    anomalous_users = [u for u, p in profiles.items() if p["is_anomalous"]]

    all_events = []

    print(f"Injecting anomaly patterns for {len(anomalous_users)} users...")
    for uid in anomalous_users:
        profile   = profiles[uid]
        generator = ANOMALY_GENERATORS[profile["anomaly_type"]]
        all_events.extend(generator(uid, profile))

    print(f"Generating remaining events to reach {N_EVENTS} total...")
    remaining = N_EVENTS - len(all_events)
    for _ in range(remaining):
        uid     = random.choice(user_ids)
        profile = profiles[uid]
        e       = generate_normal_event(uid, profile)
        if profile["is_anomalous"]:
            e["is_anomalous"] = 1
            e["anomaly_type"] = profile["anomaly_type"]
        all_events.append(e)

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_path = config["data"]["raw_path"]
    df.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Dataset saved     : {out_path}")
    print(f"Total events      : {len(df)}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Anomalous users   : {len(anomalous_users)} / {N_USERS}")
    print(f"\nEvent breakdown:")
    print(df["event_type"].value_counts())
    print(f"\nAnomaly breakdown:")
    print(df[df["is_anomalous"]==1]["anomaly_type"].value_counts())
    print(f"{'='*50}")
    return df

if __name__ == "__main__":
    generate_dataset()