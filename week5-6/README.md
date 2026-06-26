# Week 5-6: Classical Machine Learning & Anomaly Detection

## Before You Begin

Before exploring the contents of this folder, make sure you have completed the Week 5-6 learning resources provided in the main repository.

The videos are the primary learning material for these weeks and will introduce concepts such as:

* Machine Learning fundamentals
* Supervised vs Unsupervised Learning
* Anomaly Detection
* Isolation Forest
* Local Outlier Factor (LOF)

The purpose of this folder is to strengthen those concepts and help you connect them with a real-world implementation.

---

## Learning Objectives

By the end of Weeks 5-6, you should be able to:

* Understand the standard machine learning workflow
* Distinguish between supervised and unsupervised learning
* Understand what anomaly detection is and why it is important
* Learn how different anomaly detection models work
* Understand why Isolation Forest is commonly used in industry
* Read and understand a real anomaly detection implementation

---

## Folder Structure

### `sklearn_workflow.md`

This document introduces the standard machine learning pipeline used across most projects.

Topics include:

* Data collection
* Feature engineering
* Data preprocessing
* Model training
* Prediction
* Evaluation
* Deployment

Understanding this workflow will make it easier to learn any machine learning model in the future.

---

### `anomaly_detection_models.md`

This document introduces several popular anomaly detection approaches.

Models covered:

* Isolation Forest
* Local Outlier Factor (LOF)
* One-Class SVM
* Autoencoders

The goal is not to master every algorithm, but to understand the intuition behind them.

---

### `model_comparison.md`

This document provides a broader overview of machine learning models.

Topics include:

* Regression
* Classification
* Clustering
* Anomaly Detection

This should help you understand where Isolation Forest fits within the larger machine learning landscape.

---

### `isolation_forest.py`

This is a production-style implementation of an Isolation Forest anomaly detection pipeline.

The code includes:

* Feature selection
* Data preprocessing
* Standardization
* Isolation Forest training
* Anomaly scoring
* Model evaluation
* Model persistence

Do not worry if every line does not immediately make sense.

Focus on understanding the overall flow of the pipeline and how the concepts from the learning materials connect to the implementation.

---

## Suggested Learning Order

1. Complete the Week 5-6 video resources from the main repository.
2. Read `sklearn_workflow.md`
3. Read `model_comparison.md`
4. Read `anomaly_detection_models.md`
5. Study `isolation_forest.py`

---

## Questions To Think About

While reading the implementation, try answering the following:

1. What features are being given to the model?
2. Why are the features scaled before training?
3. What does the contamination parameter represent?
4. How does Isolation Forest identify anomalies?
5. What is the purpose of anomaly scores?
6. Why are evaluation metrics important?
7. What are the advantages of Isolation Forest compared to other anomaly detection methods?

---

## Deliverable

Create a short report (1–2 pages) covering:

* How Isolation Forest works
* The machine learning workflow used in the implementation
* The purpose of feature engineering
* How anomaly scores are generated
* How suspicious events are identified

The goal is not to memorize the code.

The goal is to understand how machine learning concepts translate into a real-world anomaly detection system.

Happy Learning!
