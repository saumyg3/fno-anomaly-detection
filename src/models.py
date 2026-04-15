import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import pickle
import os

# =============================================================================
# F&O Anomaly Detection — Model Training Pipeline
# =============================================================================
# Note: Full training on 2.5M+ rows requires GPU.
# Run the complete end-to-end pipeline in:
#   notebooks/FnO-Anomaly-Detection.ipynb (Google Colab with T4 GPU)
#
# This file contains the modular model definitions and training functions
# used in the notebook, structured for production readability.
# =============================================================================

# ── Isolation Forest ───────────────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray, contamination: float = 0.05) -> IsolationForest:
    """
    Train Isolation Forest — flags the top `contamination` fraction as anomalies.
    Works by isolating observations using random splits — anomalies need fewer splits.
    """
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X)
    print("  Done!")
    return model


def predict_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Returns 1 for normal, -1 for anomaly. We convert to 0/1."""
    raw = model.predict(X)
    return (raw == -1).astype(int)  # 1 = anomaly, 0 = normal


# ── LSTM Autoencoder ───────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.
    
    The idea: train the model to reconstruct normal sequences.
    Anomalies will have high reconstruction error because the model
    never learned to reconstruct unusual patterns.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder — compress sequence into latent representation
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Decoder — reconstruct sequence from latent representation
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Output layer — map back to original feature space
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)

        # Repeat hidden state for each timestep (decoder input)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode
        decoded, _ = self.decoder(decoder_input)

        # Reconstruct
        reconstructed = self.output_layer(decoded)
        return reconstructed


def create_sequences(data: np.ndarray, seq_len: int = 10) -> np.ndarray:
    """Split time series data into overlapping sequences for LSTM."""
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


def train_lstm_autoencoder(
    X: np.ndarray,
    seq_len: int = 10,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.001
) -> tuple:
    """Train LSTM Autoencoder and return model + threshold."""
    print("Training LSTM Autoencoder...")

    # Subsample for speed — 100k rows is enough to learn normal patterns
    if len(X) > 100000:
        idx = np.random.choice(len(X), 100000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    sequences = create_sequences(X_sample, seq_len)
    tensor = torch.FloatTensor(sequences)

    model = LSTMAutoencoder(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Mini-batch training
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size]
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(tensor):.6f}")

    # Compute reconstruction errors on training data to set threshold
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        errors = ((tensor - reconstructed) ** 2).mean(dim=(1, 2)).numpy()

    # Threshold = 95th percentile of reconstruction error
    threshold = np.percentile(errors, 95)
    print(f"  LSTM threshold set at: {threshold:.6f}")
    print("  Done!")
    return model, threshold, seq_len


def predict_lstm(model, X: np.ndarray, threshold: float, seq_len: int) -> np.ndarray:
    """Flag sequences with reconstruction error above threshold as anomalies."""
    model.eval()
    sequences = create_sequences(X, seq_len)
    tensor = torch.FloatTensor(sequences)

    with torch.no_grad():
        reconstructed = model(tensor)
        errors = ((tensor - reconstructed) ** 2).mean(dim=(1, 2)).numpy()

    # Map sequence-level anomalies back to row-level
    anomaly_flags = np.zeros(len(X))
    for i, error in enumerate(errors):
        if error > threshold:
            anomaly_flags[i + seq_len - 1] = 1

    return anomaly_flags.astype(int)


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def train_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> np.ndarray:
    """
    DBSCAN clustering — points that don't belong to any cluster are anomalies (label = -1).
    Subsample for speed since DBSCAN is O(n^2).
    """
    print("Running DBSCAN...")

    # Subsample — DBSCAN is slow on millions of rows
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        idx = np.arange(len(X))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_sample)

    # Map back: -1 = anomaly
    anomaly_flags = np.zeros(len(X))
    anomaly_idx = idx[labels == -1]
    anomaly_flags[anomaly_idx] = 1

    n_anomalies = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Found {n_clusters} clusters, {n_anomalies} anomalies in sample")
    print("  Done!")
    return anomaly_flags.astype(int)


# ── Ensemble ───────────────────────────────────────────────────────────────────

def ensemble_anomaly_score(if_preds, lstm_preds, dbscan_preds, weights=(0.4, 0.4, 0.2)):
    """
    Combine all 3 models into a weighted ensemble score.
    Score > 0.5 = anomaly.
    """
    score = (
        weights[0] * if_preds +
        weights[1] * lstm_preds +
        weights[2] * dbscan_preds
    )
    return score, (score >= 0.5).astype(int)


# ── Main training pipeline ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading featured data...")
    df = pd.read_parquet("data/featured.parquet")

    feature_cols = [
        "oi_change_rate", "contracts_zscore", "rolling_volatility",
        "pcr", "is_expiry_week", "value_per_contract", "oi_surge"
    ]

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0))
    print(f"  Feature matrix: {X.shape}")

    # Train all 3 models
    if_model = train_isolation_forest(X)
    lstm_model, lstm_threshold, seq_len = train_lstm_autoencoder(X)
    dbscan_preds = train_dbscan(X)

    # Get predictions
    print("\nGenerating predictions...")
    if_preds = predict_isolation_forest(if_model, X)
    lstm_preds = predict_lstm(lstm_model, X, lstm_threshold, seq_len)
    ensemble_scores, ensemble_preds = ensemble_anomaly_score(if_preds, lstm_preds, dbscan_preds)

    # Add predictions to dataframe
    df["if_anomaly"] = if_preds
    df["lstm_anomaly"] = lstm_preds
    df["dbscan_anomaly"] = dbscan_preds
    df["ensemble_score"] = ensemble_scores
    df["ensemble_anomaly"] = ensemble_preds

    # Summary
    print(f"\n── Anomaly Detection Results ──")
    print(f"Isolation Forest:  {if_preds.sum():,} anomalies ({if_preds.mean()*100:.1f}%)")
    print(f"LSTM Autoencoder:  {lstm_preds.sum():,} anomalies ({lstm_preds.mean()*100:.1f}%)")
    print(f"DBSCAN:            {dbscan_preds.sum():,} anomalies ({dbscan_preds.mean()*100:.1f}%)")
    print(f"Ensemble:          {ensemble_preds.sum():,} anomalies ({ensemble_preds.mean()*100:.1f}%)")

    # Save results
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/results.parquet", index=False)

    # Save models
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/isolation_forest.pkl", "wb") as f:
        pickle.dump(if_model, f)
    torch.save(lstm_model.state_dict(), "models/lstm_autoencoder.pt")

    print("\nAll models and results saved!")