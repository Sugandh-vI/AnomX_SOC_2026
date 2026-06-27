# LSTM Basics

LSTM stands for **Long Short-Term Memory**, a special type of Recurrent Neural Network (RNN) designed to learn patterns in sequential data.

Unlike traditional machine learning models, an LSTM processes one event at a time while remembering useful information from previous events. This allows it to learn long-term behavioral patterns.

For example, an LSTM can learn that:

* A user usually trades only during business hours.
* A user typically logs in from one country.
* Large withdrawals are usually preceded by deposits.

If future behavior differs significantly from these learned patterns, the model can recognize it as unusual.

In this project, the LSTM is used inside an Autoencoder architecture. Rather than predicting future events, it learns how to reconstruct normal behavioral sequences. Sequences that cannot be reconstructed well are treated as potential anomalies.

Understanding LSTMs is important because they form the foundation of many modern applications, including fraud detection, speech recognition, language translation, and time-series forecasting.
