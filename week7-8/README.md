# Week 7–8: Deep Learning for Sequence Modeling

Welcome to Weeks 7–8 of the AnomX Mentorship Program!

In the previous weeks, we built a complete anomaly detection pipeline using feature engineering and classical machine learning algorithms such as Isolation Forest and Local Outlier Factor. While these models work well on tabular data, they do not naturally capture how user behavior changes over time.

In this module, we move into **Deep Learning for Sequential Data**. Modern fraud detection systems rarely make decisions based on a single event. Instead, they analyze sequences of user actions to identify suspicious behavioral patterns.

During these two weeks, you will learn how sequence models can understand user activity over time and how they are used in real-world anomaly detection systems.

Recommended Learning Order: Start by watching the recommended YouTube resources to build an intuitive understanding of the concepts. Then read the markdown notes in the order provided, and finally explore the Python implementation files to see how these concepts are applied in a real-world anomaly detection system.

Note: The markdown files are intentionally kept concise and serve as quick reference notes. For a deeper understanding of the underlying concepts, we strongly recommend watching the provided YouTube resources before diving into the implementation.

## Topics Covered

* Understanding Sequential Data
* LSTM (Long Short-Term Memory) Networks
* Autoencoders for Anomaly Detection
* Real-Time Behavioral Modeling

## Learning Outcomes

By the end of this module, you should be able to:

* Understand why sequential data is different from tabular data.
* Explain how LSTMs learn temporal patterns.
* Understand how Autoencoders detect anomalies using reconstruction error.
* Read and understand an LSTM Autoencoder implementation.
* Understand how a trained model can be used to score events in real time.

## Files Included

| File                             | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| `Sequential_Data.md`             | Introduction to sequential data and why it matters.       |
| `LSTM_Basics.md`                 | Concepts behind LSTMs and sequence learning.              |
| `Autoencoders.md`                | Understanding Autoencoders and anomaly detection.         |
| `Real_Time_Behavior_Modeling.md` | Using trained models for real-time behavioral analysis.   |
| `lstm_autoencoder.py`            | Complete implementation of an LSTM Autoencoder.           |
| `forex_guard_scorer.py`          | Real-time inference pipeline for scoring incoming events. |

Take your time to understand the concepts before reading the implementation. The goal of this module is not just to run the code but to understand why sequence models are one of the most widely used approaches for behavioral anomaly detection.
