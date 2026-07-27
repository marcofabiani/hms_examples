"""Detectability: how strong must a fault be before it is seen?

The fault intensity is swept *inside* the Monte Carlo loop, so at a given
intensity the runs still scatter and detection is a random event.  The question
"from which intensity is the fault visible" therefore has no deterministic
answer; what it has is a probability of detection as a function of intensity -
the POD curve of non destructive testing, and the same machinery applies.

What is reported per failure class:

* ``i50``    : intensity detected half of the time;
* ``i90``    : intensity detected 90% of the time;
* ``i90_95`` : upper 95% confidence bound on ``i90``, i.e. the intensity you
  need before claiming 90% detection with 95% confidence.  This is the
  conservative number, always ``>= i90``.

Two estimates are produced side by side: a non parametric one read off the
binned empirical curve, and a parametric one from the classic log-odds model
``logit(POD) = b0 + b1 * log(intensity)`` (Berens, MIL-HDBK-1823), fitted by
maximum likelihood.  When they disagree, the model assumption is the suspect.

A POD curve is only meaningful **at a stated false positive rate**: lowering the
detection threshold raises POD for free.  Every function here either uses the
detector's calibrated threshold or re-calibrates it on a nominal set through
``at_alpha``, and the rate used is recorded in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .config import get_rng

__all__ = [
    "PODResult",
    "score_frame",
    "empirical_pod",
    "fit_pod",
    "pod_analysis",
    "pod_table",
    "detectability_summary",
    "sensor_sensitivity",
    "wilson_interval",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because the bins near POD = 1 have
    proportions at the boundary, where the naive interval is nonsense.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo, hi = center - half, center + half
    # the interval always contains p analytically; pin the saturated cases so
    # that rounding cannot push an endpoint the wrong side of it
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (float(min(max(0.0, lo), p)), float(max(min(1.0, hi), p)))


def _logistic_irls(
    x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-9,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """Fit ``logit(p) = b0 + b1 * x`` by iteratively reweighted least squares.

    Written out rather than delegated to scikit-learn on purpose: the sklearn
    estimator regularises by default and the way to switch that off has changed
    name across versions, which is exactly the kind of dependency this library
    avoids.  The tiny ridge here only keeps the normal equations solvable under
    complete separation; it is far too small to bias a fit that has information
    in it.

    Returns ``(beta, converged)``.
    """
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    penalty = ridge * np.eye(2)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-10, None)
        z = eta + (y - p) / w
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X + penalty, XtW @ z)
        except np.linalg.LinAlgError:
            return beta, False
        if not np.all(np.isfinite(new)):
            return beta, False
        if np.max(np.abs(new - beta)) < tol:
            return new, True
        beta = new
    return beta, False


def _invert(beta: np.ndarray, target: float) -> float:
    """Intensity at which the fitted POD equals ``target``."""
    b0, b1 = float(beta[0]), float(beta[1])
    if not np.isfinite(b0) or not np.isfinite(b1) or b1 <= 0.0:
        return float("nan")
    log_i = (np.log(target / (1.0 - target)) - b0) / b1
    if not np.isfinite(log_i) or log_i > 700.0:
        return float("nan")
    return float(np.exp(log_i))


def _empirical_crossing(bins: pd.DataFrame, target: float) -> float:
    """Smallest bin centre from which the empirical rate stays above ``target``.

    Reading the crossing from the right (the rate must stay high, not merely
    touch the level once) avoids being fooled by a single lucky bin.
    """
    if not len(bins):
        return float("nan")
    rates = bins["rate"].to_numpy()
    centers = bins["center"].to_numpy()
    crossing = float("nan")
    for i in range(len(rates) - 1, -1, -1):
        if rates[i] >= target:
            crossing = float(centers[i])
        else:
            break
    return crossing


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class PODResult:
    """POD analysis for one failure class."""

    label: str = ""
    n: int = 0
    n_detected: int = 0
    detection_rate: float = float("nan")
    intensity_min: float = float("nan")
    intensity_max: float = float("nan")
    alpha: float = float("nan")          # false positive rate the curve holds at
    threshold: float = float("nan")

    i50: float = float("nan")
    i90: float = float("nan")
    i90_95: float = float("nan")
    i50_empirical: float = float("nan")
    i90_empirical: float = float("nan")

    coef: Tuple[float, float] = (float("nan"), float("nan"))
    fitted: bool = False
    separated: bool = False
    bins: pd.DataFrame = field(default_factory=pd.DataFrame)
    boot_i90: np.ndarray = field(default_factory=lambda: np.array([]))
    notes: List[str] = field(default_factory=list)

    def pod(self, intensity: Union[float, np.ndarray]) -> np.ndarray:
        """Fitted probability of detection at the given intensity."""
        i = np.atleast_1d(np.asarray(intensity, dtype=float))
        out = np.full(i.shape, np.nan)
        if not self.fitted:
            return out
        ok = i > 0
        eta = self.coef[0] + self.coef[1] * np.log(np.where(ok, i, 1.0))
        out[ok] = 1.0 / (1.0 + np.exp(-np.clip(eta[ok], -35.0, 35.0)))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {
            "class": self.label,
            "n": self.n,
            "detected": self.n_detected,
            "detection_rate": self.detection_rate,
            "i50": self.i50,
            "i90": self.i90,
            "i90_95": self.i90_95,
            "i50_emp": self.i50_empirical,
            "i90_emp": self.i90_empirical,
            "i_max": self.intensity_max,
            "fitted": self.fitted,
            "separated": self.separated,
        }

    def summary(self) -> str:
        lines = [
            "POD - %s" % self.label,
            "  runs           : %d, detected %d (%.1f%%)"
            % (self.n, self.n_detected, 100.0 * self.detection_rate),
            "  intensity range: %.4g .. %.4g" % (self.intensity_min, self.intensity_max),
            "  at FPR         : %.3g (threshold %.4g)" % (self.alpha, self.threshold),
            "  i50 / i90      : %.4g / %.4g" % (self.i50, self.i90),
            "  i90/95         : %.4g" % self.i90_95,
            "  empirical      : i50 %.4g, i90 %.4g"
            % (self.i50_empirical, self.i90_empirical),
        ]
        if self.notes:
            lines.append("  notes:")
            lines.extend("    - " + n for n in self.notes)
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "PODResult(%r, n=%d, i90=%.4g)" % (self.label, self.n, self.i90)


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

def score_frame(
    detector: Any,
    op: Any,
    classes: Optional[Sequence[str]] = None,
    at_alpha: Optional[float] = None,
    nominal: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Tidy table ``class | intensity | score | detected`` for the failure runs.

    Parameters
    ----------
    at_alpha : re-calibrate the detector threshold on ``nominal`` (or on the
        operating point's nominal cloud) so that the analysis holds at a stated
        false positive rate.  Comparing detectors is only meaningful this way.

    The threshold used is attached to the frame as ``frame.attrs['threshold']``.
    """
    if op.intensity is None:
        raise ValueError(
            "operating point %r has no intensity column; POD analysis needs one "
            "(set columns.intensity in the manifest)" % op.name
        )
    sensors = list(detector.features_used_) or list(op.sensors)
    threshold = float(detector.threshold_)
    if at_alpha is not None:
        ref = nominal if nominal is not None else op.nominal
        saved = detector.threshold_
        detector.calibrate_threshold(ref[sensors], alpha=at_alpha)
        threshold = float(detector.threshold_)
        detector.threshold_ = saved  # leave the detector as we found it

    sub = op.failures_subset(classes)
    labels = (sub[op.label].astype(str).to_numpy() if op.label is not None
              else np.array(["(all)"] * len(sub)))
    scores = detector.score(sub[sensors])
    frame = pd.DataFrame({
        "class": labels,
        "intensity": np.abs(op.intensity_values(sub)),
        "intensity_signed": op.intensity_values(sub),
        "score": scores,
        "detected": scores > threshold,
    })
    frame.attrs["threshold"] = threshold
    frame.attrs["alpha"] = (at_alpha if at_alpha is not None
                            else float(getattr(detector, "alpha", np.nan)))
    return frame


# --------------------------------------------------------------------------
# POD
# --------------------------------------------------------------------------

def empirical_pod(
    intensity: np.ndarray,
    detected: np.ndarray,
    n_bins: int = 10,
    min_count: int = 15,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Detection rate over quantile bins of intensity, with Wilson intervals.

    Quantile bins, not equal width: a Monte Carlo sweep does not sample the
    intensity uniformly and equal width bins would leave some nearly empty.
    """
    x = np.asarray(intensity, dtype=float)
    y = np.asarray(detected, dtype=bool)
    ok = np.isfinite(x) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return pd.DataFrame(columns=["center", "lo", "hi", "n", "k", "rate",
                                     "ci_lo", "ci_hi"])

    n_bins = max(1, min(int(n_bins), max(1, x.size // max(min_count, 1))))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([x.min(), x.max() + 1e-12])
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)

    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson_interval(k, n, confidence)
        rows.append({
            "center": float(np.median(x[m])),
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": n, "k": k, "rate": k / n, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def fit_pod(
    intensity: np.ndarray,
    detected: np.ndarray,
    label: str = "",
    n_bins: int = 10,
    min_count: int = 15,
    n_boot: int = 200,
    confidence: float = 0.95,
    random_state: Any = 0,
) -> PODResult:
    """Fit the log-odds POD model and derive i50, i90 and the bound on i90.

    ``n_boot`` bootstrap refits give the sampling distribution of ``i90``; its
    upper ``confidence`` percentile is ``i90_95``.  Set ``n_boot=0`` to skip it.
    """
    x = np.asarray(intensity, dtype=float)
    y = np.asarray(detected, dtype=bool)
    res = PODResult(label=label)

    finite = np.isfinite(x)
    positive = finite & (x > 0)
    if finite.sum() and positive.sum() < finite.sum():
        res.notes.append(
            "%d run(s) with zero or missing intensity excluded from the fit"
            % int(finite.sum() - positive.sum())
        )
    x, y = x[positive], y[positive]
    res.n = int(x.size)
    res.n_detected = int(y.sum())
    if res.n == 0:
        res.notes.append("no usable run")
        return res

    res.detection_rate = res.n_detected / res.n
    res.intensity_min = float(x.min())
    res.intensity_max = float(x.max())
    res.bins = empirical_pod(x, y, n_bins=n_bins, min_count=min_count,
                             confidence=confidence)
    res.i50_empirical = _empirical_crossing(res.bins, 0.5)
    res.i90_empirical = _empirical_crossing(res.bins, 0.9)

    if res.n_detected == 0:
        res.notes.append(
            "never detected over the sampled intensity range: i90 is beyond %.4g"
            % res.intensity_max
        )
        return res
    if res.n_detected == res.n:
        res.separated = True
        res.notes.append(
            "always detected, even at the lowest sampled intensity (%.4g): the "
            "model cannot locate the transition, i90 is below the sampled range"
            % res.intensity_min
        )

    logx = np.log(x)
    beta, converged = _logistic_irls(logx, y.astype(float))
    res.coef = (float(beta[0]), float(beta[1]))
    res.fitted = bool(converged and np.isfinite(beta).all())
    if not converged:
        res.notes.append("the logistic fit did not converge")
    if res.fitted and beta[1] <= 0:
        res.notes.append(
            "detection does not increase with intensity (slope %.3g): either the "
            "sweep is too narrow or the detector is not responding" % beta[1]
        )

    if res.fitted and not res.separated:
        res.i50 = _invert(beta, 0.5)
        res.i90 = _invert(beta, 0.9)
    elif res.separated:
        res.i50 = res.i90 = float("nan")

    if n_boot and res.fitted and not res.separated:
        rng = get_rng(random_state)
        boot = []
        for _ in range(int(n_boot)):
            take = rng.randint(0, res.n, size=res.n)
            b, ok = _logistic_irls(logx[take], y[take].astype(float))
            if ok:
                value = _invert(b, 0.9)
                if np.isfinite(value):
                    boot.append(value)
        res.boot_i90 = np.asarray(boot, dtype=float)
        if res.boot_i90.size >= 20:
            res.i90_95 = float(np.quantile(res.boot_i90, confidence))
        else:
            res.notes.append(
                "bootstrap produced only %d usable refits: no confidence bound "
                "on i90" % res.boot_i90.size
            )

    if (np.isfinite(res.i90) and np.isfinite(res.intensity_max)
            and res.i90 > res.intensity_max):
        res.notes.append(
            "i90 = %.4g lies beyond the sampled range (max %.4g): it is an "
            "extrapolation, widen the sweep before trusting it"
            % (res.i90, res.intensity_max)
        )
    return res


def pod_analysis(
    detector: Any,
    op: Any,
    classes: Optional[Sequence[str]] = None,
    at_alpha: Optional[float] = None,
    nominal: Optional[pd.DataFrame] = None,
    n_bins: int = 10,
    min_count: int = 15,
    n_boot: int = 200,
    confidence: float = 0.95,
    random_state: Any = 0,
) -> Dict[str, PODResult]:
    """POD analysis of every failure class of one operating point."""
    frame = score_frame(detector, op, classes, at_alpha=at_alpha, nominal=nominal)
    threshold = frame.attrs["threshold"]
    alpha = frame.attrs["alpha"]

    results: Dict[str, PODResult] = {}
    for label, group in frame.groupby("class", sort=True):
        res = fit_pod(
            group["intensity"].to_numpy(), group["detected"].to_numpy(),
            label=str(label), n_bins=n_bins, min_count=min_count,
            n_boot=n_boot, confidence=confidence, random_state=random_state,
        )
        res.threshold = threshold
        res.alpha = alpha
        results[str(label)] = res
    return results


def pod_table(results: Dict[str, PODResult], sort_by: str = "i90") -> pd.DataFrame:
    """The headline table: one row per failure class.

    Sorted by ``i90`` descending, so the classes that need the largest fault
    before being seen - the weak spots of the monitoring system - come first.
    """
    rows = [r.to_row() for r in results.values()]
    table = pd.DataFrame(rows)
    if len(table) and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, na_position="first")
    return table.reset_index(drop=True)


def detectability_summary(
    bank: Any,
    dataset: Any,
    at_alpha: Optional[float] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`pod_analysis` over every operating point of a dataset."""
    frames = []
    for name, op in dataset.items():
        if name not in bank:
            continue
        results = pod_analysis(bank[name], op, at_alpha=at_alpha, **kwargs)
        table = pod_table(results)
        table.insert(0, "op", name)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# sensor level
# --------------------------------------------------------------------------

def sensor_sensitivity(
    op: Any,
    classes: Optional[Sequence[str]] = None,
    sensors: Optional[Sequence[str]] = None,
    high_quantile: float = 0.9,
) -> pd.DataFrame:
    """Which sensor reacts first, and how strongly, for each failure class.

    For every (class, sensor) pair, the sensor value is expressed in units of
    the nominal standard deviation and regressed on the fault intensity:

    * ``slope``    : sigmas gained per unit of intensity - the sensitivity;
    * ``dev_high`` : mean absolute deviation, in sigmas, over the top decile of
      the sweep - how far the sensor ends up;
    * ``r``        : correlation between deviation and intensity, i.e. how
      cleanly the sensor tracks the fault rather than the Monte Carlo scatter.

    Sorted by class then by ``|slope|`` descending.
    """
    if op.intensity is None:
        raise ValueError("operating point %r has no intensity column" % op.name)
    sensors = list(sensors) if sensors is not None else list(op.sensors)
    wanted = list(classes) if classes is not None else op.classes

    Xn = op.nominal[sensors].to_numpy(dtype=float)
    mu, sd = np.nanmean(Xn, axis=0), np.nanstd(Xn, axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    labels = (op.failures[op.label].astype(str).to_numpy()
              if op.label is not None else np.array(["(all)"] * len(op.failures)))
    inten_all = np.abs(op.intensity_values())
    Xf = op.failures[sensors].to_numpy(dtype=float)

    rows = []
    for cls in wanted:
        m = labels == cls
        if not m.any():
            continue
        x = inten_all[m]
        Z = (Xf[m] - mu) / sd
        ok = np.isfinite(x)
        x, Z = x[ok], Z[ok]
        if x.size < 3 or np.ptp(x) <= 0:
            continue
        thr = np.quantile(x, high_quantile)
        high = x >= thr
        xc = x - x.mean()
        denom = float(np.sum(xc * xc))
        for j, name in enumerate(sensors):
            z = Z[:, j]
            slope = float(np.sum(xc * (z - z.mean())) / denom) if denom > 0 else np.nan
            sz = float(np.std(z))
            r = (float(np.corrcoef(x, z)[0, 1]) if sz > 0 else np.nan)
            rows.append({
                "class": cls,
                "sensor": name,
                "slope": slope,
                "dev_high": float(np.mean(np.abs(z[high]))) if high.any() else np.nan,
                "r": r,
            })
    table = pd.DataFrame(rows)
    if len(table):
        table["abs_slope"] = table["slope"].abs()
        table = (table.sort_values(["class", "abs_slope"], ascending=[True, False])
                 .drop(columns="abs_slope").reset_index(drop=True))
    return table
