# Week 4: Feature Engineering & Exploratory Data Analysis (EDA)

## Overview

In Week 3, we generated a realistic synthetic event dataset containing user activities such as logins, sessions, trades, deposits, and withdrawals.

While raw event logs are useful, machine learning models cannot directly understand behavioral patterns from raw events alone. The goal of Week 4 is to transform these raw events into meaningful numerical features that capture user behavior over time.

This process is called **Feature Engineering**.

By the end of this week, you should understand:

- Why feature engineering is important
- How behavioral and temporal features are created
- How rolling statistics are used
- How anomaly-related signals can be extracted
- How to perform meaningful EDA before model building

---

# Dataset Structure

You are provided with:

```text
feature_engineering.py
features.csv
```

The script reads the raw event dataset and generates additional engineered features.

The output is stored in:

```text
features.csv
```

This file contains both:

1. Original event columns
2. Newly engineered behavioral features

---

# What Does feature_engineering.py Do?

The script creates several categories of features.

## 1. Time-Based Features

Examples:

```python
time_since_last_event_sec
time_since_last_login_sec
time_since_last_deposit_sec
```

These features help answer questions such as:

- Is the user suddenly becoming more active?
- Has the user been inactive for a long period?
- Are multiple actions happening unusually close together?

---

## 2. Rolling Window Features

Examples:

```python
roll_5_trade_vol_mean
roll_5_trade_vol_std
roll_5_pnl_mean

roll_10_trade_vol_mean
roll_10_trade_vol_std
roll_10_pnl_mean
```

Rolling features summarize recent user behavior and help capture:

- Trends
- Consistency
- Recent behavioral changes

---

## 3. Session Activity Features

Examples:

```python
roll_5_click_rate_mean
roll_10_click_rate_mean
```

These summarize recent session activity and can indicate:

- Unusually high activity
- Bot-like behavior
- Sudden changes in engagement

---

## 4. Burst Features

Examples:

```python
burst_count_5min
burst_count_30min
```

These count how many events occurred within recent time windows.

They help identify:

- Activity spikes
- Automated behavior
- Suspicious bursts of actions

---

## 5. Login Behavior Features

Examples:

```python
unique_ips_last_10_logins
unique_countries_last_10_logins
unique_devices_last_10_logins
rolling_failed_attempts_5
```

These features capture authentication behavior and are commonly used in fraud detection systems.

---

## 6. Financial Behavior Features

Examples:

```python
roll_5_deposit_sum
withdrawal_to_deposit_ratio
```

These summarize recent financial activity and help identify unusual transaction patterns.

---

## 7. Statistical Features (Z-Scores)

Examples:

```python
trade_vol_zscore
pnl_zscore
amount_zscore
session_duration_zscore
```

A z-score measures how unusual a value is compared to a user's historical behavior.

Large positive or negative z-scores often indicate outliers and potential anomalies.

---

# Understanding features.csv

The generated file contains:

- Original event data
- Time-based features
- Rolling statistics
- Behavioral metrics
- Financial indicators
- Anomaly-related scores

Each row still represents a single event.

However, every event now contains additional context about the user's recent behavior.

This additional context is what machine learning models learn from.

---

# EDA Tasks

Do not limit yourself to using only:

```python
df.describe()
```

The goal is to understand the behavior hidden inside the data.

---

## Task 1: Event Distribution

Questions:

- Which event types occur most frequently?
- Are some event types rare?
- Is the dataset balanced?

Suggested analyses:

```python
df["event_type"].value_counts()
sns.countplot(data=df, x="event_type")
```

---

## Task 2: Temporal Behavior

Analyze:

```python
time_since_last_event_sec
time_since_last_login_sec
```

Questions:

- What do their distributions look like?
- Are there extreme values?
- Are there inactivity periods?

Suggested plots:

```python
sns.histplot(...)
sns.boxplot(...)
```

---

## Task 3: Burst Activity

Analyze:

```python
burst_count_5min
burst_count_30min
```

Questions:

- What is a normal burst count?
- Which users have unusually high burst activity?
- Are there clear outliers?

---

## Task 4: Login Behavior

Analyze:

```python
unique_ips_last_10_logins
unique_countries_last_10_logins
unique_devices_last_10_logins
```

Questions:

- How frequently do users switch devices?
- How frequently do users switch countries?
- Are there suspicious login patterns?

---

## Task 5: Financial Behavior

Analyze:

```python
withdrawal_to_deposit_ratio
roll_5_deposit_sum
```

Questions:

- Which users have the largest ratios?
- Are there extreme transaction patterns?
- Are some users significantly different from others?

---

## Task 6: Z-Score Analysis

Analyze:

```python
trade_vol_zscore
amount_zscore
session_duration_zscore
```

Questions:

- How many observations exceed ±2?
- How many exceed ±3?
- Which users generate these values?

These observations are often potential anomalies.

---

## Task 7: Correlation Analysis

Compute correlations among numerical features.

Questions:

- Which features are strongly related?
- Which features appear independent?
- Are some features redundant?

Suggested visualization:

```python
sns.heatmap(df.corr(numeric_only=True))
```

---

## Challenge Task

Select 3–5 users and create behavioral profiles.

For each user analyze:

- Activity frequency
- Login behavior
- Financial behavior
- Burst activity
- Statistical outliers

Try to answer:

> Would this user appear suspicious to an anomaly detection system? Why or why not?

This exercise closely resembles real-world anomaly investigation.

---

# Expected Outcome

By the end of Week 4, you should be able to:

- Explain the purpose of each engineered feature
- Interpret behavioral metrics
- Detect unusual patterns through EDA
- Understand how raw events are transformed into model-ready features
- Build intuition for anomaly detection systems

In Week 5, these engineered features will be used to build anomaly detection models.