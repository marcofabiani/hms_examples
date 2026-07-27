"""Data quality and conditioning checks.

This module answers the questions that must be settled *before* fitting any
covariance based detector:

* are there missing or non finite values?
* are there constant sensors?
* are there duplicated or collinear sensors?
* is the covariance matrix invertible at all, and at what effective rank?

The last two are not academic.  On the SSME reference data shipped with the
project, two sensors are literally the same column (r = 1.000000) and 16 of 26
eigenvalues fall below 1e-3, so the covariance is singular and its inverse is
governed by whatever regularisation is applied rather than by the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["QualityReport", "check", "correlation_matrix", "rank_report"]


# A pair above this correlation is treated as an exact duplicate.
EXACT_DUP_THRESHOLD = 1.0 - 1e-9
# A pair above this correlation is flagged as collinear.
COLLINEAR_THRESHOLD = 0.999
# Pairs above this are merely reported as strongly correlated.
STRONG_CORR_THRESHOLD = 0.98


@dataclass
class QualityReport:
    """Outcome of :func:`check`.  Inspect ``warnings`` first."""

    n_rows: int = 0
    n_sensors: int = 0
    missing: Dict[str, int] = field(default_factory=dict)
    non_finite: Dict[str, int] = field(default_factory=dict)
    constant: List[str] = field(default_factory=list)
    duplicate_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    collinear_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    strong_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    duplicate_rows: int = 0
    condition_number: float = float("nan")
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    effective_rank: int = 0
    samples_per_sensor: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    # -- reporting --------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "QualityReport",
            "  rows x sensors      : %d x %d  (n/p = %.1f)"
            % (self.n_rows, self.n_sensors, self.samples_per_sensor),
            "  rows with NaN/inf   : %d" % sum(self.non_finite.values()),
            "  duplicated rows     : %d" % self.duplicate_rows,
            "  constant sensors    : %d %s"
            % (len(self.constant), self.constant if self.constant else ""),
            "  duplicate pairs     : %d" % len(self.duplicate_pairs),
            "  collinear pairs     : %d (|r| > %.4g)"
            % (len(self.collinear_pairs), COLLINEAR_THRESHOLD),
            "  cond(corr)          : %.3g" % self.condition_number,
            "  effective rank      : %d / %d" % (self.effective_rank, self.n_sensors),
        ]
        for a, b, r in self.duplicate_pairs:
            lines.append("    duplicate : %s == %s  (r = %.6f)" % (a, b, r))
        for a, b, r in self.collinear_pairs:
            lines.append("    collinear : %s ~ %s   (r = %.6f)" % (a, b, r))
        if self.warnings:
            lines.append("  warnings:")
            lines.extend("    ! " + w for w in self.warnings)
        else:
            lines.append("  no warnings")
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def to_frame(self) -> pd.DataFrame:
        """Per sensor table: missing counts and constant flag."""
        names = sorted(set(list(self.missing) + list(self.non_finite) + self.constant))
        return pd.DataFrame({
            "sensor": names,
            "missing": [self.missing.get(n, 0) for n in names],
            "non_finite": [self.non_finite.get(n, 0) for n in names],
            "constant": [n in self.constant for n in names],
        })

    def pairs_frame(self) -> pd.DataFrame:
        rows = []
        for kind, pairs in (
            ("duplicate", self.duplicate_pairs),
            ("collinear", self.collinear_pairs),
            ("strong", self.strong_pairs),
        ):
            for a, b, r in pairs:
                rows.append({"kind": kind, "a": a, "b": b, "r": r})
        return pd.DataFrame(rows, columns=["kind", "a", "b", "r"])

    def redundant_sensors(self) -> List[str]:
        """Sensors that can be dropped without losing information.

        For every duplicate/collinear pair the second member is proposed for
        removal, keeping the first occurrence in column order.
        """
        drop: List[str] = []
        for a, b, _ in list(self.duplicate_pairs) + list(self.collinear_pairs):
            if a not in drop and b not in drop:
                drop.append(b)
        return drop

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "QualityReport(rows=%d, sensors=%d, eff_rank=%d, warnings=%d)" % (
            self.n_rows, self.n_sensors, self.effective_rank, len(self.warnings),
        )


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Correlation matrix that tolerates constant columns (returns 0 for them)."""
    X = np.asarray(X, dtype=float)
    std = np.nanstd(X, axis=0)  # ddof=0, matched by the 1/n below so that the
    safe = np.where(std > 0, std, 1.0)  # diagonal is exactly 1 and |r| <= 1
    Z = (X - np.nanmean(X, axis=0)) / safe
    n = max(X.shape[0], 1)
    R = (Z.T @ Z) / n
    R[~np.isfinite(R)] = 0.0
    dead = std <= 0
    if np.any(dead):
        R[dead, :] = 0.0
        R[:, dead] = 0.0
        R[dead, dead] = 1.0
    return R


def rank_report(
    X: np.ndarray, rcond: float = 1e-8,
) -> Tuple[float, np.ndarray, int]:
    """Return ``(condition_number, eigenvalues_desc, effective_rank)``.

    Computed on the *correlation* matrix, so the answer does not depend on the
    physical units of the sensors.
    """
    R = correlation_matrix(X)
    eig = np.linalg.eigvalsh(R)[::-1]
    eig = np.clip(eig, 0.0, None)
    top = float(eig[0]) if eig.size else 0.0
    eff = int(np.sum(eig > rcond * max(top, 1e-300)))
    positive = eig[eig > 0]
    cond = float(top / positive[-1]) if positive.size else float("inf")
    if eff < eig.size:
        cond = float("inf")
    return cond, eig, eff


def check(
    df: pd.DataFrame,
    sensors: Optional[Sequence[str]] = None,
    rcond: float = 1e-8,
    strong_threshold: float = STRONG_CORR_THRESHOLD,
) -> QualityReport:
    """Run every quality check on the sensor columns of ``df``."""
    sensors = list(sensors) if sensors is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    sub = df[sensors].apply(pd.to_numeric, errors="coerce")
    X = sub.to_numpy(dtype=float)

    rep = QualityReport(n_rows=int(X.shape[0]), n_sensors=int(X.shape[1]))
    rep.samples_per_sensor = (
        float(X.shape[0]) / X.shape[1] if X.shape[1] else float("nan")
    )

    finite = np.isfinite(X)
    for j, name in enumerate(sensors):
        n_missing = int(np.sum(pd.isna(sub[name].to_numpy())))
        n_bad = int(np.sum(~finite[:, j]))
        if n_missing:
            rep.missing[name] = n_missing
        if n_bad:
            rep.non_finite[name] = n_bad

    rep.duplicate_rows = int(sub.duplicated().sum())

    with np.errstate(invalid="ignore"):
        std = np.nanstd(X, axis=0)
    rep.constant = [sensors[j] for j in range(len(sensors)) if not std[j] > 0]

    live = [j for j in range(len(sensors)) if std[j] > 0]
    if len(live) >= 2:
        Xl = X[:, live]
        rows_ok = np.all(np.isfinite(Xl), axis=1)
        Xl = Xl[rows_ok] if rows_ok.any() else Xl
        names = [sensors[j] for j in live]
        R = correlation_matrix(Xl)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = float(R[i, j])
                a = abs(r)
                if a >= EXACT_DUP_THRESHOLD:
                    rep.duplicate_pairs.append((names[i], names[j], r))
                elif a >= COLLINEAR_THRESHOLD:
                    rep.collinear_pairs.append((names[i], names[j], r))
                elif a >= strong_threshold:
                    rep.strong_pairs.append((names[i], names[j], r))
        rep.condition_number, rep.eigenvalues, rep.effective_rank = rank_report(Xl, rcond)

    _fill_warnings(rep)
    return rep


def _fill_warnings(rep: QualityReport) -> None:
    if rep.constant:
        rep.warnings.append(
            "%d constant sensor(s): exclude them, they carry no information and "
            "make the covariance singular" % len(rep.constant)
        )
    if rep.duplicate_pairs:
        rep.warnings.append(
            "%d exactly duplicated sensor pair(s): the covariance is singular by "
            "construction, drop one of each pair" % len(rep.duplicate_pairs)
        )
    if rep.collinear_pairs:
        rep.warnings.append(
            "%d near collinear pair(s) (|r| > %.4g): the covariance inverse will be "
            "dominated by regularisation along those directions"
            % (len(rep.collinear_pairs), COLLINEAR_THRESHOLD)
        )
    if rep.n_sensors and rep.effective_rank < rep.n_sensors:
        rep.warnings.append(
            "effective rank %d < %d sensors: use a truncated inverse (inverse='eigen') "
            "or drop the redundant sensors" % (rep.effective_rank, rep.n_sensors)
        )
    if np.isfinite(rep.samples_per_sensor) and rep.samples_per_sensor < 10:
        rep.warnings.append(
            "only %.1f samples per sensor: covariance estimates are unreliable below "
            "about 10, prefer a shrinkage estimator" % rep.samples_per_sensor
        )
    total_bad = sum(rep.non_finite.values())
    if total_bad:
        rep.warnings.append("%d non finite value(s) found" % total_bad)
    if rep.duplicate_rows:
        rep.warnings.append(
            "%d duplicated row(s): check the Monte Carlo sampling" % rep.duplicate_rows
        )
