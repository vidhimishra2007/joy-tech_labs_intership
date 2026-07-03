"""
Abstract interface all three detectors (Isolation Forest, LSTM/Telemanom,
Autoencoder) should eventually conform to, so compare_models.py can loop
over them uniformly instead of special-casing each method.

The current isolation_forest.py uses a simpler functional style
(run_isolation_forest) since IF doesn't need persistent state across
channels. When you implement LSTM and Autoencoder, consider wrapping them
in classes that follow this interface -- it'll make the three-way
comparison in evaluation/compare_models.py much cleaner.
"""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """
    Common interface for anomaly detectors.

    fit()     : train on a single channel's nominal-only training data
    predict() : return binary anomaly predictions for test data
                (windows or points, whichever the model naturally works with)
    score()   : return continuous anomaly scores (higher or lower = more
                anomalous should be documented per-implementation, since
                IF and reconstruction-error models have opposite conventions)
    """

    @abstractmethod
    def fit(self, X_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    @abstractmethod
    def score(self, X_test):
        raise NotImplementedError
