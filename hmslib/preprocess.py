"""Scaling, redundancy removal and synthetic noise."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import quality
from .config import get_rng

__all__ = ["Standardizer", "drop_redundant", "add_noise", "as_matrix"]

ArrayLike = Union[np.ndarray, pd.DataFrame]


def as_matrix(
    X: ArrayLike, columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return ``(values, column_names)`` from a DataFrame or a plain array.

    When ``columns`` is given and ``X`` is a DataFrame, the columns are selected
    *by name* and reordered, which is the only safe way to feed a fitted model
    when the caller may have reshuffled the table.  Missing columns raise.
    """
    if isinstance(X, pd.DataFrame):
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise KeyError(
                    "missing column(s) required by the model: %s" % missing
                )
            sub = X[list(columns)]
        else:
            sub = X
        names = [str(c) for c in sub.columns]
        return sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), names

    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if columns is not None and values.shape[1] != len(columns):
        raise ValueError(
            "array has %d columns but the model expects %d (%s); pass a DataFrame "
            "to match by name" % (values.shape[1], len(columns), list(columns)[:5])
        )
    names = list(columns) if columns is not None else [
        "x%d" % i for i in range(values.shape[1])
    ]
    return values, [str(n) for n in names]


class Standardizer:
    """Feature scaler with a classic and a robust variant.

    method
        ``'standard'`` : subtract the mean, divide by the standard deviation.
        ``'robust'``   : subtract the median, divide by 1.4826 * MAD, which
                         estimates the same scale as the standard deviation for
                         Gaussian data but is insensitive to contamination.
        ``'none'``     : identity, kept so that pipelines can be built uniformly.

    Zero-scale features are not divided by zero: their scale is set to 1 and
    their names collected in :attr:`degenerate_`.
    """

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "robust", "none"):
            raise ValueError("method must be 'standard', 'robust' or 'none'")
        self.method = method
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.degenerate_: List[str] = []

    def fit(self, X: ArrayLike, columns: Optional[Sequence[str]] = None) -> "Standardizer":
        values, names = as_matrix(X, columns)
        self.feature_names_ = names
        if self.method == "none":
            self.center_ = np.zeros(values.shape[1])
            self.scale_ = np.ones(values.shape[1])
            return self
        if self.method == "standard":
            center = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0)
        else:
            center = np.nanmedian(values, axis=0)
            mad = np.nanmedian(np.abs(values - center), axis=0)
            scale = 1.4826 * mad
        scale = np.asarray(scale, dtype=float)
        bad = ~(scale > 0) | ~np.isfinite(scale)
        self.degenerate_ = [names[j] for j in np.flatnonzero(bad)]
        scale[bad] = 1.0
        self.center_ = np.asarray(center, dtype=float)
        self.scale_ = scale
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        values, _ = as_matrix(X, self.feature_names_)
        return (values - self.center_) / self.scale_

    def fit_transform(self, X: ArrayLike, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.fit(X, columns).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        return np.asarray(Z, dtype=float) * self.scale_ + self.center_

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sensor": self.feature_names_,
            "center": self.center_,
            "scale": self.scale_,
        })

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.feature_names_)
        return "Standardizer(method=%r, n_features=%d)" % (self.method, n)


def drop_redundant(
    X: ArrayLike,
    sensors: Optional[Sequence[str]] = None,
    threshold: float = quality.COLLINEAR_THRESHOLD,
    keep: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Select a non redundant subset of sensors.

    Returns ``(kept, dropped)`` where ``dropped`` maps each removed sensor to
    the one it duplicates.  Constant sensors are dropped as well and mapped to
    ``'(constant)'``.

    The first sensor of each redundant group is kept, so the result depends on
    column order and is reproducible.  Pass ``keep`` to protect specific names.
    """
    values, names = as_matrix(X, sensors)
    keep_set = set(keep or [])

    with np.errstate(invalid="ignore"):
        std = np.nanstd(values, axis=0)
    dropped: Dict[str, str] = {}
    alive: List[int] = []
    for j, name in enumerate(names):
        if not std[j] > 0 and name not in keep_set:
            dropped[name] = "(constant)"
        else:
            alive.append(j)

    if len(alive) >= 2:
        rows_ok = np.all(np.isfinite(values[:, alive]), axis=1)
        sub = values[:, alive][rows_ok] if rows_ok.any() else values[:, alive]
        R = np.abs(quality.correlation_matrix(sub))
        kept_local: List[int] = []
        for i in range(len(alive)):
            name = names[alive[i]]
            redundant_with = None
            for k in kept_local:
                if R[i, k] >= threshold:
                    redundant_with = names[alive[k]]
                    break
            if redundant_with is not None and name not in keep_set:
                dropped[name] = redundant_with
            else:
                kept_local.append(i)
        alive = [alive[i] for i in kept_local]

    kept = [names[j] for j in alive]
    return kept, dropped


def add_noise(
    X: ArrayLike,
    level: float,
    mode: str = "sigma",
    random_state: Any = None,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add synthetic Gaussian measurement noise.

    mode
        ``'sigma'``    : ``level`` is a multiple of each sensor's own standard
                         deviation (or of ``reference`` when given, e.g. the
                         nominal cloud's sigma - the right choice when adding
                         noise to failure data).
        ``'relative'`` : ``level`` is a fraction of each sensor's absolute mean.
        ``'absolute'`` : ``level`` is the noise standard deviation in the
                         sensor's own physical units, identical for all sensors.

    Returns a new array; the input is not modified.
    """
    values, _ = as_matrix(X)
    if level <= 0:
        return values.copy()
    rng = get_rng(random_state)

    if mode == "sigma":
        scale = reference if reference is not None else np.nanstd(values, axis=0)
        sigma = level * np.asarray(scale, dtype=float)
    elif mode == "relative":
        sigma = level * np.abs(np.nanmean(values, axis=0))
    elif mode == "absolute":
        sigma = np.full(values.shape[1], float(level))
    else:
        raise ValueError("mode must be 'sigma', 'relative' or 'absolute'")

    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return values + rng.normal(0.0, 1.0, size=values.shape) * sigma
