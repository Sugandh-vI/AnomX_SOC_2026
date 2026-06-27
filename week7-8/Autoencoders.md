# Autoencoders for Anomaly Detection

An Autoencoder is a neural network that learns to compress data into a smaller representation and then reconstruct it back to its original form.

It consists of two main parts:

* **Encoder:** Compresses the input into a compact representation.
* **Decoder:** Reconstructs the original input from that representation.

When trained on normal data, the Autoencoder becomes very good at reconstructing normal behavior.

However, when it encounters unusual or abnormal behavior, the reconstruction becomes poor, resulting in a high reconstruction error.

This reconstruction error is used as the anomaly score.

In AnomX, the model is trained primarily on normal behavioral sequences. During inference, each sequence is reconstructed, and its reconstruction error is calculated. Higher errors indicate that the observed behavior differs significantly from what the model learned during training.

This makes Autoencoders particularly useful for anomaly detection problems where abnormal examples are rare or unavailable.
