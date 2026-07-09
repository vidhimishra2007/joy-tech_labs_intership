"""
convert_smap_to_backend_csv.py

Converts real NASA SMAP/MSL channels into the backend_telemetry_history.csv
format expected by train_backend_telemetry_model.py.

Usage:
    python3 convert_smap_to_backend_csv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Path to the extracted SMAP/MSL dataset folder 
SMAP_DATA_DIR = Path("archive/data/data/train")

# Map your 4 backend features to 4 real SMAP/MSL channels
CHANNEL_MAP = {
    "battery_voltage": "P-1",
    "temperature": "S-1",
    "cpu_usage": "E-1",
    "signal_strength": "E-2",
}

SATELLITE_ID = "SAT-001"
OUTPUT_PATH = Path("data/backend_telemetry_history.csv")


def main():
    data = {}
    min_len = None

    for feature, ch in CHANNEL_MAP.items():
        arr = np.load(SMAP_DATA_DIR / f"{ch}.npy")[:, 0]  # column 0 = raw telemetry value
        data[feature] = arr
        min_len = len(arr) if min_len is None else min(min_len, len(arr))
        print(f"Loaded {ch}.npy -> {feature} ({len(arr)} rows)")

    # trim all channels to the same length
    for k in data:
        data[k] = data[k][:min_len]

    df = pd.DataFrame(data)
    df.insert(0, "satellite_id", SATELLITE_ID)
    df.insert(0, "timestamp", pd.date_range("2026-07-01", periods=min_len, freq="min"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()