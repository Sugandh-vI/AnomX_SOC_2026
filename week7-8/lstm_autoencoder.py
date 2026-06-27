# src/models/lstm_autoencoder.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CFG = config["model"]["lstm_autoencoder"]

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "login_success", "failed_attempts", "timezone_gap_hours",
    "lot_size", "trade_volume", "pnl", "margin_used", "trade_duration_seconds",
    "trade_volume_vs_baseline", "is_night_trade",
    "amount", "is_immediate_withdrawal",
    "session_duration_mins", "page_clicks", "click_rate_per_min",
    "account_age_days",
    "time_since_last_event_sec", "time_since_last_login_sec",
    "time_since_last_deposit_sec",
    "roll_5_trade_vol_mean",  "roll_5_trade_vol_std",  "roll_5_pnl_mean",
    "roll_10_trade_vol_mean", "roll_10_trade_vol_std", "roll_10_pnl_mean",
    "roll_30_trade_vol_mean", "roll_30_trade_vol_std", "roll_30_pnl_mean",
    "roll_5_click_rate_mean", "roll_10_click_rate_mean", "roll_30_click_rate_mean",
    "burst_count_5min", "burst_count_30min",
    "unique_ips_last_10_logins", "unique_countries_last_10_logins",
    "unique_devices_last_10_logins", "rolling_failed_attempts_5",
    "roll_5_deposit_sum", "withdrawal_to_deposit_ratio",
    "trade_vol_zscore", "pnl_zscore", "amount_zscore", "session_duration_zscore",
]

SEQ_LEN    = CFG["sequence_length"]   # 10
INPUT_DIM  = len(FEATURE_COLS)        # 46


# ── Model architecture ────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    Encoder compresses the input sequence into a latent vector.
    Decoder reconstructs the sequence from the latent vector.
    Anomaly score = mean squared reconstruction error per sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: input_dim → hidden_dim → latent_dim
        self.encoder = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2,
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent_dim → hidden_dim → input_dim
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → reconstructed: same shape"""
        batch_size = x.size(0)

        # Encode
        _, (h_n, _) = self.encoder(x)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        latent = self.enc_to_latent(h_n[-1])          # (batch, latent_dim)

        # Decode — repeat latent vector across seq_len timesteps
        dec_input = self.latent_to_dec(latent)         # (batch, hidden_dim)
        dec_input = dec_input.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, hidden_dim)

        dec_out, _ = self.decoder(dec_input)           # (batch, seq_len, hidden_dim)
        reconstructed = self.output_layer(dec_out)     # (batch, seq_len, input_dim)
        return reconstructed


# ── Data preparation ──────────────────────────────────────────────────────────

def make_sequences(df: pd.DataFrame, seq_len: int, only_normal: bool = False) -> tuple:
    """
    Build sliding window sequences per user.
    Returns (sequences_array, labels_array, index_array).
    index_array[i] = row index of the LAST event in sequence i.
    """
    logger.info(f"  Building sequences (seq_len={seq_len}, only_normal={only_normal})...")

    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    sequences, labels, indices = [], [], []

    for uid, group in df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        orig_idx = group.index.tolist()

        X = group[FEATURE_COLS].values.astype(np.float32)
        y = group["is_anomalous"].values

        for i in range(len(group) - seq_len + 1):
            seq   = X[i : i + seq_len]
            label = y[i + seq_len - 1]      # label of the last event in sequence

            if only_normal and label == 1:
                continue

            sequences.append(seq)
            labels.append(label)
            indices.append(orig_idx[i + seq_len - 1])

    sequences = np.array(sequences, dtype=np.float32)
    labels    = np.array(labels,    dtype=np.int32)
    indices   = np.array(indices,   dtype=np.int64)

    logger.info(f"  Sequences: {sequences.shape}, anomaly rate: {labels.mean():.3f}")
    return sequences, labels, indices


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(sequences: np.ndarray) -> tuple[LSTMAutoencoder, StandardScaler]:
    logger.info("Training LSTM Autoencoder...")

    # Scale features
    n, s, f = sequences.shape
    flat    = sequences.reshape(-1, f)
    scaler  = StandardScaler()
    flat_sc = scaler.fit_transform(flat)
    seqs_sc = flat_sc.reshape(n, s, f).astype(np.float32)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Using device: {device}")

    model = LSTMAutoencoder(
        input_dim  = INPUT_DIM,
        hidden_dim = CFG["hidden_dim"],
        latent_dim = CFG["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    tensor   = torch.tensor(seqs_sc)
    dataset  = TensorDataset(tensor)
    loader   = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True)

    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(tensor)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{CFG['epochs']} | Loss: {epoch_loss:.6f}")

    model.load_state_dict(best_state)
    logger.info(f"  Best loss: {best_loss:.6f}")
    return model, scaler


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(model: LSTMAutoencoder,
                   scaler: StandardScaler,
                   sequences: np.ndarray,
                   batch_size: int = 512) -> np.ndarray:
    """Return per-sequence reconstruction error (MSE)."""
    n, s, f = sequences.shape
    flat    = sequences.reshape(-1, f)
    flat_sc = scaler.transform(flat)
    seqs_sc = flat_sc.reshape(n, s, f).astype(np.float32)

    device  = next(model.parameters()).device
    model.eval()

    errors = []
    for i in range(0, n, batch_size):
        batch = torch.tensor(seqs_sc[i : i + batch_size]).to(device)
        recon = model(batch)
        mse   = ((recon - batch) ** 2).mean(dim=(1, 2))  # per-sequence MSE
        errors.append(mse.cpu().numpy())

    return np.concatenate(errors)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold_pct: float):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, average_precision_score)

    threshold = np.percentile(scores, threshold_pct)
    preds     = (scores >= threshold).astype(int)

    print("\n" + "="*55)
    print("  LSTM AUTOENCODER — EVALUATION REPORT")
    print("="*55)
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-"*42)
    print(f"{'Total sequences':<30} {len(y_true):>10,}")
    print(f"{'True anomalies':<30} {y_true.sum():>10,}")
    print(f"{'Threshold (pct)':<30} {threshold_pct:>10.1f}")
    print(f"{'Threshold (score)':<30} {threshold:>10.4f}")
    print(f"{'Predicted anomalies':<30} {preds.sum():>10,}")

    try:
        auc = roc_auc_score(y_true, scores)
        ap  = average_precision_score(y_true, scores)
        print(f"{'ROC-AUC':<30} {auc:>10.4f}")
        print(f"{'Average Precision':<30} {ap:>10.4f}")
    except Exception as e:
        logger.warning(f"AUC error: {e}")

    print("\nClassification Report:")
    print(classification_report(y_true, preds,
                                  target_names=["Normal", "Anomaly"],
                                  zero_division=0))
    cm = confusion_matrix(y_true, preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred Normal  Pred Anomaly")
    print(f"  Actual Normal   {cm[0,0]:>8,}      {cm[0,1]:>8,}")
    print(f"  Actual Anomaly  {cm[1,0]:>8,}      {cm[1,1]:>8,}")
    print("="*55)
    return threshold


def save_artifacts(model, scaler, threshold, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(path) / "lstm_autoencoder.pt")
    joblib.dump(scaler,    Path(path) / "lstm_scaler.pkl")
    joblib.dump(threshold, Path(path) / "lstm_threshold.pkl")
    logger.info(f"LSTM artifacts saved to {path}")


def load_artifacts(path):
    model = LSTMAutoencoder(
        input_dim  = INPUT_DIM,
        hidden_dim = CFG["hidden_dim"],
        latent_dim = CFG["latent_dim"],
    )
    model.load_state_dict(torch.load(Path(path) / "lstm_autoencoder.pt",
                                      map_location="cpu"))
    model.eval()
    scaler    = joblib.load(Path(path) / "lstm_scaler.pkl")
    threshold = joblib.load(Path(path) / "lstm_threshold.pkl")
    return model, scaler, threshold


# ── Pipeline entry ────────────────────────────────────────────────────────────

def run_lstm_autoencoder():
    processed_path = config["data"]["processed_path"]
    model_path     = config["api"]["model_path"]

    logger.info(f"Loading features from {processed_path}...")
    df = pd.read_csv(processed_path)

    # Build sequences — train ONLY on normal events
    train_seqs, train_labels, train_idx = make_sequences(
        df, SEQ_LEN, only_normal=True
    )

    # Build ALL sequences for evaluation (normal + anomalous)
    all_seqs, all_labels, all_idx = make_sequences(
        df, SEQ_LEN, only_normal=False
    )

    # Train
    model, scaler = train_model(train_seqs)

    # Score everything
    logger.info("Computing anomaly scores on full dataset...")
    scores    = compute_scores(model, scaler, all_seqs)
    threshold = evaluate(all_labels, scores,
                         CFG["threshold_percentile"])

    # Save
    save_artifacts(model, scaler, threshold, model_path)

    # Attach scores back to dataframe
    score_map = {}
    for idx, score in zip(all_idx, scores):
        score_map[idx] = max(score_map.get(idx, 0), float(score))

    df["lstm_score"]      = df.index.map(score_map).fillna(0)
    df["lstm_prediction"] = (df["lstm_score"] >= threshold).astype(int)
    df.to_csv(processed_path.replace("features.csv", "features_with_scores.csv"), index=False)

    logger.info("LSTM pipeline complete.")
    return model, scaler, threshold, df


if __name__ == "__main__":
    run_lstm_autoencoder()