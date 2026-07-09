from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "battery_voltage",
    "temperature",
    "cpu_usage",
    "signal_strength",
]

WINDOW_SIZE = 120
STRIDE = 1


def create_window_features(arr: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    windows = []

    for start in range(0, len(arr) - window_size + 1, stride):
        window = arr[start:start + window_size]
        features = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window[-1] - window[0],
        ])
        windows.append(features)

    return np.asarray(windows, dtype=float)


def train_global_model(
    csv_path: str | Path,
    output_path: str | Path,
    contamination: float = 0.05,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> dict:
    feature_names = list(feature_names)
    df = pd.read_csv(csv_path)

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {missing}")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    X_raw = df[feature_names].to_numpy(dtype=float)
    if len(X_raw) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} rows, got {len(X_raw)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_train = create_window_features(X_scaled)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "raw_feature_count": len(feature_names),
        "model_feature_count": X_train.shape[1],
        "contamination": contamination,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


class BackendTelemetryScorer:
    def __init__(self, model_path: str | Path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.window_size = artifact["window_size"]
        self.buffers = defaultdict(lambda: deque(maxlen=self.window_size))

    def handle_packet(self, satellite_id: str, telemetry: dict) -> dict:
        values = [telemetry[name] for name in self.feature_names]
        self.buffers[satellite_id].append(values)

        if len(self.buffers[satellite_id]) < self.window_size:
            return {
                "satellite_id": satellite_id,
                "status": "warming_up",
                "buffer_size": len(self.buffers[satellite_id]),
                "required_buffer_size": self.window_size,
                "prediction": None,
            }

        raw_buffer = np.asarray(self.buffers[satellite_id], dtype=float)
        scaled_buffer = self.scaler.transform(raw_buffer)
        X = create_window_features(scaled_buffer, window_size=self.window_size, stride=1)

        label = self.model.predict(X)[0]
        score = self.model.decision_function(X)[0]

        return {
            "satellite_id": satellite_id,
            "status": "scored",
            "is_anomaly": bool(label == -1),
            "label": int(label),
            "score": float(score),
            "n_samples": 1,
        }
