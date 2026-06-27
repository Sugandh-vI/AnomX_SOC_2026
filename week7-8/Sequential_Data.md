# Understanding Sequential Data

Most machine learning models assume that every data point is independent. However, in many real-world applications, the order of events carries important information.

This type of data is known as **Sequential Data**.

Examples include:

* User login history
* Financial transactions
* Stock market prices
* Sensor readings
* Website clickstreams

Consider two users:

**User A**

```
Login → Trade → Logout
```

**User B**

```
Login → 20 Failed Logins → Password Reset → Large Withdrawal
```

Although both users performed similar actions, the order and timing of those actions are completely different. Sequence models are designed to capture these temporal relationships.

In anomaly detection, suspicious behavior often develops gradually rather than appearing in a single event. By analyzing a sequence of activities instead of individual records, deep learning models can detect patterns that traditional machine learning algorithms may miss.

In AnomX, every user's recent activities are grouped into short sequences before being passed to the deep learning model for analysis.
