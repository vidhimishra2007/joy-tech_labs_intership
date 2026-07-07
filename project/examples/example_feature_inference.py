"""Smoke test for the current CERT-SAT model artifact.

This example uses feature-level inference because the included artifact does
not contain fitted scalers for raw telemetry preprocessing.
"""

import json

import numpy as np

from src.models.isolation_forest import MultiChannelIsolationForest


MODEL_PATH = "models_saved/isoforest_all_channels.joblib"
CHANNEL_ID = "P-1"


def main():
    manager = MultiChannelIsolationForest.load(MODEL_PATH)
    detector = manager.detectors[CHANNEL_ID].model
    n_features = detector.n_features_in_

    # Replace this with real scaled/windowed summary features from backend.
    X_window_features = np.zeros((1, n_features), dtype=float)
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
