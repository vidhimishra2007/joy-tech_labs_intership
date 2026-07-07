"""Smoke test for raw telemetry inference with scaler-included artifact."""

import json

import numpy as np

from src.inference.live_window import preprocess_live_window
from src.models.isolation_forest import MultiChannelIsolationForest
from src.utils.config import WINDOW_SIZE


MODEL_PATH = "models_saved/isoforest_all_channels_with_scalers.joblib"
CHANNEL_ID = "P-1"


def main():
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    scaler = manager.scalers[CHANNEL_ID]
    n_raw_features = scaler.n_features_in_

    # Replace this with the latest 120 raw telemetry packets for this channel.
    raw_buffer = np.zeros((WINDOW_SIZE, n_raw_features), dtype=float)
    X_window_features = preprocess_live_window(CHANNEL_ID, raw_buffer, manager)
    result = manager.predict(CHANNEL_ID, X_window_features)

    serializable = {
        "ch_id": result["ch_id"],
        "is_anomaly": result["is_anomaly"].astype(bool).tolist(),
        "label": result["label"].astype(int).tolist(),
        "score": result["score"].astype(float).tolist(),
        "n_samples": result["n_samples"],
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
