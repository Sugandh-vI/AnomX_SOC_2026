# AnomX – Synthetic Event Dataset Generator

Welcome to the first code release of the AnomX Mentorship Program.

This repository contains the code used to generate a synthetic financial activity dataset that will be used throughout the project for anomaly detection experiments.

The goal of this file is not to teach Machine Learning yet.

Instead, focus on understanding:

- How synthetic datasets are created
- How user behavior can be simulated
- How suspicious activities can be represented in data
- How real-world anomaly detection problems are formulated

---

# Why Do We Need Synthetic Data?

In real financial institutions, access to transaction data is highly restricted due to privacy, legal, and security concerns.

As students, we usually do not have access to such datasets.

To overcome this problem, we generate realistic synthetic data that mimics the behavior of users on a trading platform.

This allows us to:

- Build anomaly detection models
- Perform exploratory data analysis
- Create feature engineering pipelines
- Simulate fraud detection scenarios
- Experiment safely without sensitive information

---

# High-Level Workflow

The dataset generation process follows four major steps:

1. Create user profiles
2. Generate normal user activity
3. Inject anomalous behavior patterns
4. Export everything into a CSV dataset

The overall flow looks like:

User Profiles
      ↓
Normal Events
      ↓
Anomaly Injection
      ↓
Final Dataset
      ↓
events.csv

---

# Step 1: User Profile Generation

Function:

```python
build_user_profiles()
```

This function creates synthetic users.

Each user is assigned several characteristics such as:

- User ID
- Home country
- Home IP address
- Preferred device
- Trading instruments
- Typical trade volume
- Typical deposit amount
- Account creation date

Example:

```text
USER_0001
Country: IN
Device: chrome_win
Typical Deposit: 1200
Typical Trade Volume: 15000
```

These profiles represent the normal baseline behavior of users.

---

# Step 2: Normal Event Generation

Function:

```python
generate_normal_event()
```

After user profiles are created, events are generated.

Each event represents an action performed by a user.

Possible event types:

- Login
- Trade
- Deposit
- Withdrawal
- Session
- KYC Change

Every event receives:

- Timestamp
- User ID
- Event Type
- Relevant event attributes

Examples:

Login Event:

```text
USER_0001 logged in from India using Chrome
```

Trade Event:

```text
USER_0001 traded EURUSD
Volume = 15000
PnL = 120
```

Deposit Event:

```text
USER_0001 deposited ₹5000
```

---

# Step 3: Anomaly Injection

A dataset containing only normal behavior is not useful for anomaly detection.

We therefore intentionally inject suspicious patterns.

Certain users are marked as anomalous.

These users generate abnormal events that simulate fraud, abuse, or suspicious trading activity.

---

# Implemented Anomaly Types

## 1. IP Hopper

Rapid switching between countries and IP addresses.

Example:

```text
India
↓
Singapore
↓
Germany
↓
Brazil
```

within a very short time period.

Potential indicator:

Impossible travel behaviour.

---

## 2. Wash Trader

Characteristics:

- Extremely high trade volume
- Same instrument repeatedly traded
- Consistently profitable trades
- Very short trade durations

Potential indicator:

Market manipulation.

---

## 3. Deposit Withdrawal Cycler

Pattern:

Deposit
↓
Immediate Withdrawal
↓
Deposit
↓
Immediate Withdrawal

Potential indicator:

Money laundering behaviour.

---

## 4. Bot Trader

Characteristics:

- Extremely high click rates
- Very short sessions
- Activity during unusual hours

Potential indicator:

Automated scripts or bots.

---

## 5. Structurer

Characteristics:

Many deposits intentionally kept below a threshold.

Example:

```text
₹980
₹995
₹990
₹985
```

instead of:

```text
₹4000
```

Potential indicator:

Transaction structuring.

---

## 6. Brute Forcer

Characteristics:

Multiple failed login attempts followed by a successful login.

Potential indicator:

Account takeover attempt.

---

## 7. Dormant Withdrawer

Characteristics:

Long period of inactivity followed by a large withdrawal.

Potential indicator:

Compromised account.

---

## 8. Consistent Winner

Characteristics:

- Always profitable trades
- Extremely short holding times
- Very large trade volumes

Potential indicator:

Latency exploitation or suspicious trading strategies.

---

## 9. Device Switcher

Characteristics:

Different device fingerprints for every login.

Potential indicator:

Credential sharing or account compromise.

---

## 10. KYC Manipulator

Characteristics:

KYC details changed shortly before a large withdrawal.

Potential indicator:

Fraudulent identity modifications.

---

# Important Dataset Columns

Some columns worth examining:

| Column | Description |
|----------|-------------|
| user_id | Unique user identifier |
| event_type | Type of activity |
| timestamp | Event timestamp |
| ip_address | User IP address |
| country | User location |
| device | Device used |
| trade_volume | Trade amount |
| pnl | Profit and loss |
| amount | Deposit/withdrawal amount |
| failed_attempts | Failed login count |
| click_rate_per_min | Session click intensity |
| is_anomalous | Target label |
| anomaly_type | Type of anomaly |

---

# Dataset Output

Running:

```python
generate_dataset()
```

produces:

```text
events.csv
```

which contains:

- Normal users
- Suspicious users
- Multiple event types
- Behavioural signals
- Temporal information

This dataset will be used throughout the mentorship program.

---

## Included Files

I have shared two files with you:

1. `generate_events.py`
2. `events.csv`

The CSV file was generated using the code provided in `generate_events.py`.

As you go through the code, I encourage you to keep the dataset open alongside it and observe how different parts of the code translate into actual rows and columns in the dataset.

A useful exercise is to pick a few rows from the CSV and try to trace which section of the code may have generated them. This will help you understand not only Python syntax, but also the thought process behind designing synthetic datasets for anomaly detection systems.



# Suggested Exercises

Try answering the following questions using the generated dataset.

### Beginner

1. Count total events per event type.
2. Count total anomalous users.
3. Find the most common countries.
4. Find the most common devices.

### Intermediate

1. Compare normal vs anomalous trade volumes.
2. Compare login behaviour of normal and anomalous users.
3. Analyze withdrawal amounts.
4. Investigate click rates.

### Advanced

1. Build visualizations for anomaly patterns.
2. Design new behavioural features.
3. Create your own anomaly type.
4. Train a simple anomaly detection model.

---

# Things To Observe While Reading The Code

Do not try to memorize every line.

Instead focus on understanding:

1. How data is generated
2. How user profiles are created
3. How events are constructed
4. How anomalies differ from normal behaviour
5. How everything is combined into a final dataset

If you can explain the complete flow from:

User Profile → Event Generation → Anomaly Injection → CSV Dataset

then you have understood the core idea behind this module.

---

# Final Note

This file represents the first stage of a real anomaly detection system.

Before we can build Machine Learning models, we first need data.

Understanding how data is generated and how suspicious behaviour manifests itself is one of the most important skills in applied Machine Learning.

Take your time reading the code and exploring the dataset.

See you in the next session.

— Sugandh Kumar
AnomX Mentorship Program