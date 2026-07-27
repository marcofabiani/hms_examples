"""Common contract for anomaly detectors.

Every detector follows the same protocol::

    det.fit(X_nominal)          # nominal cloud of one operating point
    s = det.score(X)            # higher = more anomalous
    det.threshold_              # calibrated on nominal data only
    det.predict(X)              # boolean, True = anomalous

so that Mahalanobis, one-class SVM, PCA/SPE and the autoencoder are
interchangeable inside a pipeline and can be compared at equal false positive
rate.  Scores are always oriented "higher = more anomalous", whatever the
underlying sign convention of the algorithm.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..preprocess import as_matrix

__all__ = ["BaseDetector"]

ArrayLike = Union[np.ndarray, pd.DataFrame]


class BaseDetector:
    """Base class: fitting and scoring are left to subclasses."""

    #: filled by ``fit``: sensor names seen at fit time
    features_in_: List[str]
    #: filled by ``fit``: sensors actually used (after dropping redundant ones)
    features_used_: List[str]
    #: calibrated decision threshold
    threshold_: float

    def __init__(self) -> None:
        self.features_in_ = []
        self.features_used_ = []
        self.threshold_ = float("inf")
        self.thresholds_: Dict[str, float] = {}
        self.warnings_: List[str] = []
        self.diagnostics_: Dict[str, Any] = {}
        self._fitted = False

    # -- to implement -----------------------------------------------------
    def fit(self, X: ArrayLike, y: Any = None) -> "BaseDetector":
        raise NotImplementedError

    def score(self, X: ArrayLike) -> np.ndarray:
        """Anomaly score, higher meaning more anomalous."""
        raise NotImplementedError

    # -- shared -----------------------------------------------------------
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Boolean array, ``True`` where the sample is flagged as anomalous."""
        return self.score(X) > self.threshold_

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).predict(X)

    def set_threshold(self, value: float, name: str = "manual") -> "BaseDetector":
        self.threshold_ = float(value)
        self.thresholds_[name] = float(value)
        self.diagnostics_["threshold_rule"] = name
        return self

    def use_threshold(self, name: str) -> "BaseDetector":
        """Switch to one of the thresholds computed at fit time."""
        if name not in self.thresholds_:
            raise KeyError(
                "unknown threshold %r; available: %s" % (name, list(self.thresholds_))
            )
        self.threshold_ = float(self.thresholds_[name])
        self.diagnostics_["threshold_rule"] = name
        return self

    def calibrate_threshold(
        self, X_nominal: ArrayLike, alpha: float = 1e-3, name: str = "empirical",
    ) -> "BaseDetector":
        """Set the threshold to the ``1 - alpha`` quantile of nominal scores.

        ``X_nominal`` must be data the detector was *not* fitted on, otherwise
        the false positive rate is optimistic.
        """
        s = np.asarray(self.score(X_nominal), dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            raise ValueError("no finite score on the calibration set")
        if s.size * alpha < 10:
            self.warnings_.append(
                "empirical threshold at alpha=%.3g estimated from %d samples: the "
                "quantile rests on ~%.1f points, prefer a chi2 threshold or more data"
                % (alpha, s.size, s.size * alpha)
            )
        value = float(np.quantile(s, 1.0 - alpha))
        self.thresholds_[name] = value
        self.threshold_ = value
        self.diagnostics_["threshold_rule"] = name
        self.diagnostics_["calibration_samples"] = int(s.size)
        return self

    def false_positive_rate(self, X_nominal: ArrayLike) -> float:
        """Fraction of nominal samples flagged as anomalous."""
        return float(np.mean(self.predict(X_nominal)))

    # -- input handling ---------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("%s is not fitted" % type(self).__name__)

    def _prepare_fit(
        self, X: ArrayLike, sensors: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        values, names = as_matrix(X, sensors)
        self.features_in_ = list(names)
        self.features_used_ = list(names)
        return values

    def _prepare(self, X: ArrayLike) -> np.ndarray:
        """Select and order the columns the model was fitted on."""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            values, _ = as_matrix(X, self.features_used_)
            return values
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] == len(self.features_used_):
            return values
        if values.shape[1] == len(self.features_in_):
            idx = [self.features_in_.index(c) for c in self.features_used_]
            return values[:, idx]
        raise ValueError(
            "array has %d columns; the model expects %d (used) or %d (as fitted). "
            "Pass a DataFrame to match columns by name."
            % (values.shape[1], len(self.features_used_), len(self.features_in_))
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> str:
        """Persist with joblib, next to a readable JSON metadata file."""
        import joblib

        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        joblib.dump(self, path)
        meta = {
            "class": type(self).__name__,
            "module": type(self).__module__,
            "features_in": self.features_in_,
            "features_used": self.features_used_,
            "threshold": float(self.threshold_),
            "thresholds": {k: float(v) for k, v in self.thresholds_.items()},
            "diagnostics": _jsonable(self.diagnostics_),
            "warnings": list(self.warnings_),
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> "BaseDetector":
        import joblib

        return joblib.load(path)

    # -- reporting --------------------------------------------------------
    def summary(self) -> str:
        lines = ["%s" % type(self).__name__]
        if self._fitted:
            lines.append("  sensors used : %d / %d"
                         % (len(self.features_used_), len(self.features_in_)))
            lines.append("  threshold    : %.6g  (%s)"
                         % (self.threshold_, self.diagnostics_.get("threshold_rule", "?")))
            for name, value in self.thresholds_.items():
                lines.append("     %-12s %.6g" % (name, value))
            for key, value in self.diagnostics_.items():
                if key in ("threshold_rule", "spectrum"):
                    continue
                lines.append("  %-13s %s" % (key, _fmt(value)))
        else:
            lines.append("  not fitted")
        if self.warnings_:
            lines.append("  warnings:")
            lines.extend("    ! " + w for w in self.warnings_)
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._fitted else "unfitted"
        return "%s(%s)" % (type(self).__name__, state)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, np.ndarray):
        return "array(shape=%s)" % (value.shape,)
    return str(value)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
