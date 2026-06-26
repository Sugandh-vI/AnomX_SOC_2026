# Common Anomaly Detection Models

Anomaly detection focuses on identifying observations that behave differently from the majority of the data.

In real-world systems, anomalies often represent:

* Fraud
* Security threats
* System failures
* Suspicious user behavior
* Unusual transactions

There is no single best anomaly detection algorithm.

Different algorithms detect anomalies in different ways.

---

# 1. Isolation Forest

## Core Idea

Anomalies are easier to isolate than normal points.

Imagine repeatedly splitting data using random rules.

Example:

Trade Volume < 500

Login Count < 10

Withdrawal Amount < 1000

An anomalous point usually gets separated very quickly.

Normal points require many splits before becoming isolated.

---

## Advantages

* Fast
* Scales well
* Works with high-dimensional data
* Industry standard

---

## Limitations

* May miss local anomalies
* Less interpretable than simpler methods

---

# 2. Local Outlier Factor (LOF)

## Core Idea

Compare a point's density with its neighbors.

If a point is significantly less dense than nearby points, it is considered anomalous.

---

## Example

Imagine a crowded city.

One house located far away from all other houses appears unusual.

LOF measures this local isolation.

---

## Advantages

* Detects local anomalies well
* Good for clustered datasets

---

## Limitations

* Slower than Isolation Forest
* Computationally expensive on large datasets

---

# 3. One-Class SVM

## Core Idea

Learn the boundary around normal data.

Anything outside the boundary is considered anomalous.

---

## Advantages

* Strong mathematical foundation
* Effective on smaller datasets

---

## Limitations

* Slow
* Difficult to tune
* Does not scale well

---

# 4. Autoencoders

## Core Idea

Train a neural network to reconstruct normal behavior.

Normal events reconstruct well.

Anomalous events reconstruct poorly.

Large reconstruction error indicates anomaly.

---

## Advantages

* Extremely powerful
* Captures complex patterns

---

## Limitations

* Requires more data
* More difficult to train
* Less interpretable

---

# Which Model Should You Use?

There is no universal answer.

A common progression is:

Isolation Forest
→ LOF
→ Autoencoders

This progression moves from simple and interpretable approaches toward more powerful deep learning methods.

This is also the path followed in the AnomX project.
