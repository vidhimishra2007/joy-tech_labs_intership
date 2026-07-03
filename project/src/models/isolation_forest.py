"""
Isolation Forest baseline detector for CERT-SAT.

Trains one IsolationForest per channel (SMAP/MSL channels have very
different behavior, so a single global model would underperform), using
summary-statistic windows as input features.
"""

from sklearn.ensemble import IsolationForest

from src.data.preprocessing import scale_channel
from src.data.windowing import create_windows_stats
from src.data.loader import load_channel_arrays
from src.utils.config import (
    IF_N_ESTIMATORS, IF_RANDOM_STATE,
    IF_DEFAULT_CONTAMINATION, IF_CONTAMINATION_MIN, IF_CONTAMINATION_MAX,
)


def run_isolation_forest(labels, shape_df, train_dir, test_dir,
                          window_size, stride, contam_lookup=None):
    """
    Fit + predict an IsolationForest per channel.

    Parameters
    ----------
    labels : DataFrame from load_labels()
    shape_df : DataFrame from build_shape_summary(), used to skip channels
        too short for the chosen window_size
    contam_lookup : dict {ch_id: anomaly_fraction}, e.g. from
        build_imbalance_summary(). If None, IF_DEFAULT_CONTAMINATION is used
        for every channel (leakage-free baseline). If provided, contamination
        is set per-channel from true test anomaly fraction -- useful as an
        "oracle upper bound" comparison, but note this leaks test labels
        into a hyperparameter; report this choice explicitly.

    Returns
    -------
    results : list of dicts, one per successfully modeled channel, with
        keys: ch_id, spacecraft, class, preds, scores, anomaly_sequences, test_len.
        Matches the format expected by src.evaluation.metrics.aggregate_results.
    skipped_channels : list of ch_ids skipped for being too short.
    """
    contam_lookup = contam_lookup or {}
    results = []
    skipped_channels = []

    for _, row in labels.iterrows():
        ch_id = row["chan_id"]

        match = shape_df[shape_df["ch_id"] == ch_id]
        if match.empty or match["train_len"].values[0] < window_size or match["test_len"].values[0] < window_size:
            skipped_channels.append(ch_id)
            continue

        train_arr, test_arr = load_channel_arrays(ch_id, train_dir, test_dir)
        if train_arr is None:
            skipped_channels.append(ch_id)
            continue

        train_scaled, test_scaled = scale_channel(train_arr, test_arr)
        X_train = create_windows_stats(train_scaled, window_size, stride)
        X_test = create_windows_stats(test_scaled, window_size, stride)

        if len(X_train) == 0 or len(X_test) == 0:
            skipped_channels.append(ch_id)
            continue

        contam_rate = contam_lookup.get(ch_id, IF_DEFAULT_CONTAMINATION)
        contam_rate = min(max(contam_rate, IF_CONTAMINATION_MIN), IF_CONTAMINATION_MAX)

        clf = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=contam_rate,
            random_state=IF_RANDOM_STATE,
        )
        clf.fit(X_train)

        preds = clf.predict(X_test)
        scores = clf.decision_function(X_test)

        results.append({
            "ch_id": ch_id,
            "spacecraft": row["spacecraft"],
            "class": row["class"],
            "preds": preds,
            "scores": scores,
            "anomaly_sequences": row["anomaly_sequences"],
            "test_len": test_arr.shape[0],
        })

    return results, skipped_channels
