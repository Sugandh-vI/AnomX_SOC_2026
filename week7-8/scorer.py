import numpy as np
import torch
import joblib
import yaml
from collections import deque
from pathlib import Path
from src.models.lstm_autoencoder import LSTMAutoencoder, FEATURE_COLS, INPUT_DIM
from src.utils.logger import get_logger

logger = get_logger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

CFG        = config["model"]["lstm_autoencoder"]
MODEL_PATH = config["api"]["model_path"]
SEQ_LEN    = CFG["sequence_length"]


# ---------------- SAFE VALUE ----------------
def safe_value(x):
    if x is None:
        return 0.0
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x
    except:
        return 0.0


class ForexGuardScorer:

    def __init__(self, model_path: str = MODEL_PATH):
        logger.info("Loading ForexGuard models...")
        self._load_lstm(model_path)
        self.user_buffers: dict[str, deque] = {}
        logger.info("ForexGuard scorer ready.")

    def _load_lstm(self, path: str):
        self.lstm_model = LSTMAutoencoder(
            input_dim  = INPUT_DIM,
            hidden_dim = CFG["hidden_dim"],
            latent_dim = CFG["latent_dim"],
        )

        self.lstm_model.load_state_dict(
            torch.load(Path(path) / "lstm_autoencoder.pt", map_location="cpu")
        )
        self.lstm_model.eval()

        self.lstm_scaler    = joblib.load(Path(path) / "lstm_scaler.pkl")
        self.lstm_threshold = joblib.load(Path(path) / "lstm_threshold.pkl")

        logger.info(f"LSTM loaded (threshold={self.lstm_threshold:.4f})")

    # ---------------- FEATURE EXTRACTION ----------------
    def _extract_features(self, event: dict) -> np.ndarray:
        values = []

        for col in FEATURE_COLS:
            val = event.get(col, 0.0)
            values.append(safe_value(val))   # 🔥 FIX

        return np.array(values, dtype=np.float32)

    # ---------------- LSTM SCORING ----------------
    def _score_lstm(self, user_id: str, feat_vec: np.ndarray) -> float:
        if user_id not in self.user_buffers:
            self.user_buffers[user_id] = deque(maxlen=SEQ_LEN)

        self.user_buffers[user_id].append(feat_vec)

        if len(self.user_buffers[user_id]) < SEQ_LEN:
            return 0.0

        seq = np.array(list(self.user_buffers[user_id]), dtype=np.float32)

        flat   = seq.reshape(-1, INPUT_DIM)
        flat_s = self.lstm_scaler.transform(flat)

        # 🔥 CRITICAL FIX
        flat_s = np.nan_to_num(flat_s, nan=0.0, posinf=0.0, neginf=0.0)

        seq_s  = flat_s.reshape(1, SEQ_LEN, INPUT_DIM)

        tensor = torch.tensor(seq_s)

        with torch.no_grad():
            recon = self.lstm_model(tensor)
            mse   = float(((recon - tensor) ** 2).mean().item())

        return safe_value(mse)

    # ---------------- SEVERITY ----------------
    def _severity(self, score: float) -> str:
        if score > 6:
            return "CRITICAL"
        elif score > 4:
            return "HIGH"
        elif score > 2:
            return "MEDIUM"
        else:
            return "LOW"

    # ---------------- EXPLANATION ----------------
    def _explain(self, top_features):
        explanations = []

        for f in top_features:
            name = f["feature"]

            if name == "amount":
                explanations.append("Unusual transaction amount")
            elif name == "withdrawal_to_deposit_ratio":
                explanations.append("Abnormal withdrawal behavior")
            elif name == "pnl":
                explanations.append("Unexpected profit/loss pattern")
            elif name == "login_success":
                explanations.append("Suspicious login pattern")
            elif name == "failed_attempts":
                explanations.append("Multiple failed login attempts")
            elif name == "burst_count_5min":
                explanations.append("High activity burst detected")

        if not explanations:
            explanations.append("Behavior deviates from historical pattern")

        return explanations[:2]

    # ---------------- TOP FEATURES ----------------
    def _top_features(self, feat_vec: np.ndarray, n: int = 5):
        scaled = self.lstm_scaler.transform(feat_vec.reshape(1, -1))[0]

        # 🔥 CRITICAL FIX
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        top_idx = np.argsort(np.abs(scaled))[::-1][:n]

        return [
            {
                "feature": FEATURE_COLS[i],
                "raw_value": safe_value(feat_vec[i]),
                "scaled_value": safe_value(scaled[i]),
            }
            for i in top_idx
        ]

    # ---------------- MAIN SCORE ----------------
    def score(self, event: dict) -> dict:

        user_id = event.get("user_id", "UNKNOWN")
        feat_vec = self._extract_features(event)

        lstm_error = self._score_lstm(user_id, feat_vec)

        is_anomaly = lstm_error >= (self.lstm_threshold * 1.5)

        top_feats = self._top_features(feat_vec)
        severity = self._severity(lstm_error)
        reasons = self._explain(top_feats)

        return {
            "user_id": str(user_id),
            "event_id": str(event.get("event_id", "")),
            "event_type": str(event.get("event_type", "")),
            "timestamp": str(event.get("timestamp", "")),

            "lstm_score": safe_value(lstm_error),
            "anomaly_score": safe_value(lstm_error),
            "is_anomaly": bool(is_anomaly),

            "severity": str(severity),
            "reasons": [str(r) for r in reasons],
            "verdict": "🚨 ANOMALY" if bool(is_anomaly) else "✅ NORMAL",

            "top_features": top_feats,
        }