"""Mahalanobis distance detector, built to survive ill conditioned covariances.

The distance itself is elementary; everything that can go wrong lives in the
inverse covariance.  On engine data the sensors are redundant by construction
(derived quantities, duplicated measurements), so the covariance is routinely
singular or nearly so, and a naive ``inv`` produces distances that are governed
by numerical noise rather than by physics.

The chain implemented here is explicit and inspectable at every step:

1. structural pre-check: constant and redundant sensors are removed *before*
   estimating anything (:func:`hmslib.preprocess.drop_redundant`);
2. scaling, classic or robust;
3. covariance estimation with a selectable estimator;
4. factorisation, never an explicit inverse: either a Cholesky factor of
   ``Sigma + lambda*I`` or a truncated eigendecomposition.  Both produce a
   matrix ``A`` with ``d^2 = ||A (x - mu)||^2``, which is numerically stable and
   makes the effective rank explicit;
5. diagnostics (condition number, effective rank, ridge actually used, n/p);
6. three comparable thresholds: chi-square, Hotelling T^2 with the finite
   sample F correction, and an out-of-sample empirical quantile.

``contributions`` decomposes the squared distance over sensors, which is what
turns detection into isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import stats

from ..preprocess import Standardizer, as_matrix, drop_redundant
from ..config import get_rng
from .base import BaseDetector

__all__ = ["Mahalanobis"]

ArrayLike = Union[np.ndarray, pd.DataFrame]

_COV_ESTIMATORS = ("empirical", "ledoit_wolf", "oas", "mcd", "diagonal")
_INVERSE_MODES = ("cholesky", "eigen")
_THRESHOLD_RULES = ("empirical", "chi2", "hotelling")


class Mahalanobis(BaseDetector):
    """Mahalanobis distance from the nominal cloud of one operating point.

    Parameters
    ----------
    cov : {'empirical', 'ledoit_wolf', 'oas', 'mcd', 'diagonal'}
        Covariance estimator.  ``'ledoit_wolf'`` shrinks towards a scaled
        identity and is the safe default when sensors are many or correlated.
        ``'mcd'`` is robust to contamination of the nominal set (use it with
        ``scaler='robust'``); it requires more samples than sensors.
    scaler : {'standard', 'robust', 'none'}
        Feature scaling applied before estimating the covariance.
    inverse : {'cholesky', 'eigen'}
        ``'cholesky'`` factorises ``Sigma + ridge*I``; ``'eigen'`` truncates the
        spectrum, giving a pseudo-inverse of explicit rank.  Use ``'eigen'``
        when the covariance is rank deficient and you would rather ignore the
        null directions than regularise them.
    ridge : float or 'auto' or None
        Ridge added to the diagonal for ``inverse='cholesky'``.  ``'auto'``
        starts at ``rcond * trace(Sigma) / p`` and is increased tenfold until
        the factorisation succeeds; the value actually used is reported in
        ``diagnostics_['ridge']``.
    rcond : float
        Relative floor on the eigenvalues, used both for the automatic ridge and
        for the rank truncation of ``inverse='eigen'``.
    var_explained : float, optional
        Alternative truncation rule for ``inverse='eigen'``: keep the leading
        components explaining this fraction of the variance (e.g. ``0.99``).
    drop_redundant_sensors : bool
        Remove constant and collinear sensors before fitting.  Leave it on: an
        exactly duplicated sensor makes the covariance singular by construction.
    collinear_threshold : float
        ``|r|`` above which two sensors are considered redundant.
    threshold : {'empirical', 'chi2', 'hotelling'}
        Rule used to set ``threshold_``.  All the rules that can be computed are
        stored in ``thresholds_`` so they can be compared.
    alpha : float
        Target false positive rate on nominal data.
    calib_fraction : float
        Fraction of the nominal set held out to calibrate the empirical
        threshold.  In-sample distances are biased low, so the held-out
        quantile is the honest choice; set to 0 to disable (then the empirical
        threshold is not available).
    """

    def __init__(
        self,
        cov: str = "ledoit_wolf",
        scaler: str = "standard",
        inverse: str = "cholesky",
        ridge: Union[float, str, None] = "auto",
        rcond: float = 1e-8,
        var_explained: Optional[float] = None,
        drop_redundant_sensors: bool = True,
        collinear_threshold: float = 0.999,
        threshold: str = "empirical",
        alpha: float = 1e-3,
        calib_fraction: float = 0.3,
        support_fraction: Optional[float] = None,
        random_state: Any = 0,
    ) -> None:
        super().__init__()
        if cov not in _COV_ESTIMATORS:
            raise ValueError("cov must be one of %s" % (_COV_ESTIMATORS,))
        if inverse not in _INVERSE_MODES:
            raise ValueError("inverse must be one of %s" % (_INVERSE_MODES,))
        if threshold not in _THRESHOLD_RULES:
            raise ValueError("threshold must be one of %s" % (_THRESHOLD_RULES,))
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= calib_fraction < 1.0:
            raise ValueError("calib_fraction must lie in [0, 1)")

        self.cov = cov
        self.scaler = scaler
        self.inverse = inverse
        self.ridge = ridge
        self.rcond = float(rcond)
        self.var_explained = var_explained
        self.drop_redundant_sensors = bool(drop_redundant_sensors)
        self.collinear_threshold = float(collinear_threshold)
        self.threshold = threshold
        self.alpha = float(alpha)
        self.calib_fraction = float(calib_fraction)
        self.support_fraction = support_fraction
        self.random_state = random_state

        # fitted state
        self.scaler_: Optional[Standardizer] = None
        self.location_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.dropped_sensors_: Dict[str, str] = {}
        self._A: Optional[np.ndarray] = None
        self._precision: Optional[np.ndarray] = None
        self.n_components_: int = 0

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(
        self, X: ArrayLike, y: Any = None, sensors: Optional[Sequence[str]] = None,
    ) -> "Mahalanobis":
        """Fit on the nominal cloud of a single operating point."""
        values, names = as_matrix(X, sensors)
        self.features_in_ = list(names)

        if self.drop_redundant_sensors:
            kept, dropped = drop_redundant(
                values, names, threshold=self.collinear_threshold
            )
            self.dropped_sensors_ = dropped
            if not kept:
                raise ValueError("every sensor was dropped as constant or redundant")
            if dropped:
                self.warnings_.append(
                    "%d sensor(s) dropped as redundant: %s"
                    % (len(dropped), ", ".join("%s~%s" % (k, v) for k, v in dropped.items()))
                )
        else:
            kept, self.dropped_sensors_ = list(names), {}
        self.features_used_ = kept
        idx = [names.index(c) for c in kept]
        data = values[:, idx]

        rows_ok = np.all(np.isfinite(data), axis=1)
        if not np.all(rows_ok):
            self.warnings_.append(
                "%d row(s) with non finite values ignored while fitting"
                % int((~rows_ok).sum())
            )
            data = data[rows_ok]
        if data.shape[0] < 2:
            raise ValueError("at least 2 usable nominal samples are required")

        train, calib = self._split(data)

        self.scaler_ = Standardizer(self.scaler).fit(train, kept)
        Ztr = self.scaler_.transform(pd.DataFrame(train, columns=kept))

        self.location_, self.covariance_ = self._estimate(Ztr)
        self._A, self.n_components_, ridge_used, spectrum = self._factorize(self.covariance_)
        self._precision = self._A.T @ self._A

        self._fill_diagnostics(Ztr, ridge_used, spectrum)
        self._fitted = True
        self._compute_thresholds(train, calib, spectrum)
        return self

    def _split(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.calib_fraction <= 0.0:
            return data, None
        n_calib = int(round(self.calib_fraction * data.shape[0]))
        if n_calib < 2 or data.shape[0] - n_calib < 2:
            self.warnings_.append(
                "nominal set too small to hold out a calibration split "
                "(%d samples): the empirical threshold will be in-sample and "
                "therefore optimistic" % data.shape[0]
            )
            return data, data
        perm = get_rng(self.random_state).permutation(data.shape[0])
        return data[perm[n_calib:]], data[perm[:n_calib]]

    def _estimate(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, p = Z.shape
        if self.cov == "diagonal":
            return np.mean(Z, axis=0), np.diag(np.var(Z, axis=0, ddof=1))
        if self.cov == "mcd" and n <= p:
            self.warnings_.append(
                "MinCovDet needs more samples than sensors (%d <= %d): "
                "falling back to Ledoit-Wolf" % (n, p)
            )
            estimator_name = "ledoit_wolf"
        else:
            estimator_name = self.cov

        from sklearn.covariance import OAS, EmpiricalCovariance, LedoitWolf, MinCovDet

        if estimator_name == "empirical":
            est: Any = EmpiricalCovariance(store_precision=False)
        elif estimator_name == "ledoit_wolf":
            est = LedoitWolf(store_precision=False)
        elif estimator_name == "oas":
            est = OAS(store_precision=False)
        else:
            est = MinCovDet(
                support_fraction=self.support_fraction,
                random_state=get_rng(self.random_state),
            )
            if self.scaler != "robust":
                self.warnings_.append(
                    "cov='mcd' with scaler=%r: the scaling is not robust, so "
                    "outliers still influence it; scaler='robust' is more "
                    "consistent" % self.scaler
                )
        est.fit(Z)
        return np.asarray(est.location_, dtype=float), np.asarray(est.covariance_, dtype=float)

    def _factorize(
        self, cov: np.ndarray,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """Return ``(A, n_components, ridge_used, spectrum)`` with d^2 = ||A(x-mu)||^2."""
        cov = 0.5 * (cov + cov.T)  # enforce symmetry against round-off
        p = cov.shape[0]
        spectrum = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)

        if self.inverse == "eigen":
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
            top = float(evals[0]) if evals.size else 0.0
            if self.var_explained is not None:
                total = float(evals.sum())
                if total <= 0:
                    raise ValueError("degenerate covariance: total variance is zero")
                k = int(np.searchsorted(np.cumsum(evals) / total, self.var_explained) + 1)
                k = int(np.clip(k, 1, p))
            else:
                k = int(np.sum(evals > self.rcond * max(top, 1e-300)))
                k = max(k, 1)
            if k < p:
                self.warnings_.append(
                    "truncated inverse: %d of %d directions kept, the remaining "
                    "ones are treated as non informative" % (k, p)
                )
            A = (evecs[:, :k] / np.sqrt(evals[:k])).T
            return A, k, 0.0, spectrum

        # cholesky
        trace_scale = float(np.trace(cov)) / max(p, 1)
        if self.ridge is None:
            lam = 0.0
        elif isinstance(self.ridge, str):
            if self.ridge != "auto":
                raise ValueError("ridge must be a float, 'auto' or None")
            lam = self.rcond * max(trace_scale, 1e-300)
        else:
            lam = float(self.ridge)

        eye = np.eye(p)
        L = None
        for attempt in range(14):
            try:
                L = np.linalg.cholesky(cov + lam * eye)
                break
            except np.linalg.LinAlgError:
                lam = max(lam * 10.0, self.rcond * max(trace_scale, 1e-300)) or 1e-12
        if L is None:
            raise np.linalg.LinAlgError(
                "covariance could not be factorised even with a ridge of %.3g; "
                "use inverse='eigen'" % lam
            )
        if lam > 1e-6 * max(trace_scale, 1e-300):
            self.warnings_.append(
                "a ridge of %.3g (%.2g x mean variance) was needed to factorise the "
                "covariance: the distance along the weakest directions is set by "
                "this regularisation, not by the data"
                % (lam, lam / max(trace_scale, 1e-300))
            )
        A = sla.solve_triangular(L, eye, lower=True)
        return A, p, lam, spectrum

    def _fill_diagnostics(
        self, Ztr: np.ndarray, ridge_used: float, spectrum: np.ndarray,
    ) -> None:
        n, p = Ztr.shape
        positive = spectrum[spectrum > 0]
        top = float(spectrum[0]) if spectrum.size else 0.0
        eff_rank = int(np.sum(spectrum > self.rcond * max(top, 1e-300)))
        cond = (float(top / positive[-1]) if positive.size and eff_rank == spectrum.size
                else float("inf"))
        self.diagnostics_.update({
            "n_train": int(n),
            "n_sensors_in": len(self.features_in_),
            "n_sensors_used": p,
            "samples_per_sensor": float(n) / p if p else float("nan"),
            "cov_estimator": self.cov,
            "inverse": self.inverse,
            "ridge": float(ridge_used),
            "n_components": int(self.n_components_),
            "effective_rank": eff_rank,
            "condition_number": cond,
            "spectrum": spectrum,
        })
        if n < 10 * p:
            self.warnings_.append(
                "only %.1f samples per sensor: the covariance estimate is shaky "
                "below about 10, prefer cov='ledoit_wolf' or fewer sensors"
                % (float(n) / p)
            )
        if eff_rank < p:
            self.warnings_.append(
                "the nominal covariance has effective rank %d over %d sensors: "
                "%d direction(s) carry no nominal variance" % (eff_rank, p, p - eff_rank)
            )

    def _compute_thresholds(
        self, train: np.ndarray, calib: Optional[np.ndarray], spectrum: np.ndarray,
    ) -> None:
        p_eff = int(self.n_components_)
        n = int(self.diagnostics_["n_train"])

        self.thresholds_["chi2"] = float(
            np.sqrt(stats.chi2.ppf(1.0 - self.alpha, df=max(p_eff, 1)))
        )
        if n > p_eff + 1:
            f_crit = stats.f.ppf(1.0 - self.alpha, p_eff, n - p_eff)
            t2 = p_eff * (n + 1.0) * (n - 1.0) / (n * (n - p_eff)) * f_crit
            self.thresholds_["hotelling"] = float(np.sqrt(t2))
        if calib is not None:
            frame = pd.DataFrame(calib, columns=self.features_used_)
            s = np.asarray(self.score(frame), dtype=float)
            s = s[np.isfinite(s)]
            if s.size:
                if s.size * self.alpha < 10:
                    self.warnings_.append(
                        "empirical threshold at alpha=%.3g rests on ~%.1f calibration "
                        "points: it is noisy, compare it with the chi2 value"
                        % (self.alpha, s.size * self.alpha)
                    )
                self.thresholds_["empirical"] = float(np.quantile(s, 1.0 - self.alpha))
                self.diagnostics_["calibration_samples"] = int(s.size)

        rule = self.threshold
        if rule not in self.thresholds_:
            fallback = "chi2" if "chi2" in self.thresholds_ else list(self.thresholds_)[0]
            self.warnings_.append(
                "threshold rule %r unavailable, using %r instead" % (rule, fallback)
            )
            rule = fallback
        if self.cov != "empirical" and rule == "hotelling":
            self.warnings_.append(
                "the Hotelling correction assumes the sample covariance; with "
                "cov=%r it is only approximate" % self.cov
            )
        self.threshold_ = float(self.thresholds_[rule])
        self.diagnostics_["threshold_rule"] = rule

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _delta(self, X: ArrayLike) -> np.ndarray:
        values = self._prepare(X)
        Z = self.scaler_.transform(pd.DataFrame(values, columns=self.features_used_))
        return Z - self.location_

    def squared_distance(self, X: ArrayLike) -> np.ndarray:
        """Squared Mahalanobis distance, ``d^2``."""
        delta = self._delta(X)
        proj = delta @ self._A.T
        return np.einsum("ij,ij->i", proj, proj)

    def score(self, X: ArrayLike) -> np.ndarray:
        """Mahalanobis distance ``d`` (higher = more anomalous).

        The distance is returned rather than its square so that the value keeps
        its "number of sigmas" reading.
        """
        return np.sqrt(np.clip(self.squared_distance(X), 0.0, None))

    def contributions(
        self, X: ArrayLike, normalize: bool = False, as_frame: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Per sensor decomposition of ``d^2``.

        ``c_i = (x - mu)_i * [Sigma^-1 (x - mu)]_i``, which sums exactly to the
        squared distance.  Individual terms can be negative when correlated
        sensors deviate in a way the nominal correlation does not expect; that
        is informative, not a bug.  With ``normalize=True`` each row is divided
        by its ``d^2``, giving the share of the distance carried by each sensor.
        """
        delta = self._delta(X)
        contrib = delta * (delta @ self._precision)
        if normalize:
            total = contrib.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                contrib = np.where(np.abs(total) > 0, contrib / total, 0.0)
        if not as_frame:
            return contrib
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(contrib, columns=self.features_used_, index=index)

    def top_contributors(self, X: ArrayLike, k: int = 5) -> pd.DataFrame:
        """The ``k`` sensors carrying most of the distance, per sample."""
        contrib = self.contributions(X, normalize=True)
        order = np.argsort(-contrib.to_numpy(), axis=1)[:, :k]
        cols = np.asarray(self.features_used_)
        data = {}
        for j in range(order.shape[1]):
            data["sensor_%d" % (j + 1)] = cols[order[:, j]]
            data["share_%d" % (j + 1)] = contrib.to_numpy()[
                np.arange(len(contrib)), order[:, j]
            ]
        return pd.DataFrame(data, index=contrib.index)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    @property
    def precision_(self) -> np.ndarray:
        """Inverse covariance actually used (in scaled space)."""
        self._check_fitted()
        return self._precision

    def threshold_table(self) -> pd.DataFrame:
        """The available thresholds side by side, with the implied alpha."""
        self._check_fitted()
        rows = []
        for name, value in self.thresholds_.items():
            rows.append({
                "rule": name,
                "threshold": value,
                "selected": name == self.diagnostics_.get("threshold_rule"),
            })
        return pd.DataFrame(rows)
