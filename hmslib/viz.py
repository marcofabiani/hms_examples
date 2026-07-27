"""Plotting helpers.

Every function returns a matplotlib ``Figure`` and never calls ``show``, so the
same code works in a notebook, in a script and inside a multipage PDF report.
Only matplotlib is used - no seaborn, which is not guaranteed to exist on the
target machine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .compat import categorical_colors
from .config import PLOT, get_rng

__all__ = [
    "ellipse_points",
    "binned_stats",
    "plot_sensor_histograms",
    "plot_correlation_heatmap",
    "plot_eigen_spectrum",
    "plot_scatter_2d",
    "plot_pca",
    "trend_vs_intensity",
    "plot_score_distributions",
    "plot_contributions",
    "plot_class_counts",
    "plot_pod",
    "plot_score_vs_intensity",
    "plot_sensor_sensitivity",
    "plot_contribution_heatmap",
]


def _subsample(n: int, max_points: Optional[int] = None, seed: int = 0) -> np.ndarray:
    max_points = PLOT.max_scatter_points if max_points is None else max_points
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.sort(get_rng(seed).choice(n, size=max_points, replace=False))


def _class_colors(classes: Sequence[str]) -> Dict[str, Any]:
    return dict(zip(classes, categorical_colors(len(classes))))


def ellipse_points(
    mean: np.ndarray, cov: np.ndarray, radius: float = 3.0, n_points: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trace the ``radius``-sigma ellipse of a 2-D Gaussian."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    evals, evecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    transform = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None))) * float(radius)
    pts = np.asarray(mean, dtype=float).reshape(2, 1) + transform @ circle
    return pts[0], pts[1]


def binned_stats(
    x: np.ndarray, y: np.ndarray, n_bins: int = 12, min_count: int = 5,
) -> pd.DataFrame:
    """Median and interquartile band of ``y`` over quantile bins of ``x``.

    Quantile bins rather than equal width bins: with a Monte Carlo sweep the
    intensity is not uniformly sampled, and equal width bins would leave some of
    them nearly empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return pd.DataFrame(columns=["center", "lo", "hi", "median", "q25", "q75", "n"])

    n_bins = max(1, min(int(n_bins), max(1, x.size // max(min_count, 1))))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if edges.size < 2:
        edges = np.array([x.min(), x.max() + 1e-12])
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)

    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        if not np.any(m):
            continue
        yb = y[m]
        rows.append({
            "center": float(np.median(x[m])),
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "median": float(np.median(yb)),
            "q25": float(np.percentile(yb, 25)),
            "q75": float(np.percentile(yb, 75)),
            "n": int(m.sum()),
        })
    return pd.DataFrame(rows)


def _grid(n: int, ncols: int, width: float = 2.6, height: float = 2.0):
    import matplotlib.pyplot as plt

    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# --------------------------------------------------------------------------
# distributions and structure
# --------------------------------------------------------------------------

def plot_sensor_histograms(
    df: pd.DataFrame,
    sensors: Optional[Sequence[str]] = None,
    standardize: bool = True,
    ncols: int = 6,
    bins: int = 40,
    title: Optional[str] = None,
):
    """Histogram of every sensor, with a N(0,1) reference when standardized."""
    sensors = list(sensors) if sensors is not None else list(df.columns)
    X = df[sensors].to_numpy(dtype=float)
    if standardize:
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)

    fig, axes = _grid(len(sensors), ncols)
    ref_x = np.linspace(-4.5, 4.5, 200)
    ref_y = np.exp(-0.5 * ref_x ** 2) / np.sqrt(2.0 * np.pi)
    for ax, name, j in zip(axes, sensors, range(len(sensors))):
        col = X[:, j]
        col = col[np.isfinite(col)]
        ax.hist(col, bins=bins, density=True, color="#4878a8", alpha=0.8, edgecolor="none")
        if standardize:
            ax.plot(ref_x, ref_y, "r--", lw=1.0)
            ax.set_xlim(-5, 5)
        ax.set_title(str(name), fontsize=PLOT.font_size - 1)
        ax.set_yticks([])
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    sensors: Optional[Sequence[str]] = None,
    title: str = "Sensor correlation",
    annotate_threshold: Optional[float] = 0.98,
):
    """Correlation matrix; pairs above ``annotate_threshold`` are marked."""
    import matplotlib.pyplot as plt

    sensors = list(sensors) if sensors is not None else list(df.columns)
    from .quality import correlation_matrix

    R = correlation_matrix(df[sensors].to_numpy(dtype=float))
    n = len(sensors)
    size = max(4.0, min(0.28 * n + 2.0, 14.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sensors, rotation=90, fontsize=max(4.0, 90.0 / max(n, 1)))
    ax.set_yticklabels(sensors, fontsize=max(4.0, 90.0 / max(n, 1)))
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    if annotate_threshold is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(R[i, j]) >= annotate_threshold:
                    ax.plot(j, i, marker="s", ms=3, mfc="none", mec="k", mew=0.7)
    fig.tight_layout()
    return fig


def plot_eigen_spectrum(
    df: pd.DataFrame,
    sensors: Optional[Sequence[str]] = None,
    rcond: float = 1e-8,
    title: str = "Correlation eigenvalue spectrum",
):
    """Eigenvalues of the correlation matrix on a log scale.

    The flat tail at the bottom is exactly the part of the space where the
    inverse covariance is decided by regularisation rather than by data.
    """
    import matplotlib.pyplot as plt

    from .quality import rank_report

    sensors = list(sensors) if sensors is not None else list(df.columns)
    cond, eig, eff = rank_report(df[sensors].to_numpy(dtype=float), rcond)
    floor = rcond * (eig[0] if eig.size else 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    k = np.arange(1, eig.size + 1)
    ax.semilogy(k, np.clip(eig, 1e-20, None), "o-", ms=3.5, color="#20486e")
    ax.axhline(floor, color=PLOT.threshold_color, ls="--", lw=1.0,
               label="rcond floor (%.0e)" % rcond)
    ax.axvline(eff + 0.5, color="#666666", ls=":", lw=1.0,
               label="effective rank = %d / %d" % (eff, eig.size))
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.set_title("%s  -  cond = %.3g" % (title, cond))
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# sensor space views
# --------------------------------------------------------------------------

def plot_scatter_2d(
    op: Any,
    sx: str,
    sy: str,
    classes: Optional[Sequence[str]] = None,
    sigma: float = 3.0,
    max_points: Optional[int] = None,
):
    """Two sensors against each other: nominal cloud, sigma ellipse, failures."""
    import matplotlib.pyplot as plt

    nom = op.nominal[[sx, sy]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    idx = _subsample(len(nom), max_points)
    ax.scatter(nom[idx, 0], nom[idx, 1], s=PLOT.scatter_size, marker="s",
               c=PLOT.nominal_color, alpha=0.5, edgecolors="none",
               label="nominal", zorder=1)

    if op.label is not None and len(op.failures):
        wanted = list(classes) if classes is not None else op.classes
        colors = _class_colors(wanted)
        labels = op.failures[op.label].astype(str).to_numpy()
        for cls in wanted:
            m = labels == cls
            if not m.any():
                continue
            xy = op.failures.loc[m, [sx, sy]].to_numpy(dtype=float)
            sel = _subsample(len(xy), max_points)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=PLOT.scatter_size,
                       color=colors[cls], alpha=PLOT.scatter_alpha,
                       edgecolors="none", label=cls, zorder=2)

    mu = np.nanmean(nom, axis=0)
    cov = np.cov(nom, rowvar=False)
    ex, ey = ellipse_points(mu, cov, radius=sigma)
    ax.plot(ex, ey, "k--", lw=1.3, label=r"%g$\sigma$ ellipse" % sigma, zorder=3)

    ax.set_xlabel(sx)
    ax.set_ylabel(sy)
    ax.set_title("%s - %s vs %s" % (op.name, sy, sx))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=PLOT.font_size - 2)
    fig.tight_layout()
    return fig


def plot_pca(
    op: Any,
    classes: Optional[Sequence[str]] = None,
    n_components: int = 2,
    max_points: Optional[int] = None,
):
    """PCA fitted on the nominal cloud, failures projected onto it.

    Fitting on the nominal only (rather than on everything) keeps the axes
    interpretable as directions of nominal Monte Carlo scatter.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    sensors = op.sensors
    Xn = op.nominal[sensors].to_numpy(dtype=float)
    mu, sd = np.nanmean(Xn, axis=0), np.nanstd(Xn, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Zn = (Xn - mu) / sd

    pca = PCA(n_components=min(n_components, Zn.shape[1]))
    Pn = pca.fit_transform(Zn)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    idx = _subsample(len(Pn), max_points)
    ax.scatter(Pn[idx, 0], Pn[idx, 1], s=PLOT.scatter_size, marker="s",
               c=PLOT.nominal_color, alpha=0.5, edgecolors="none", label="nominal")

    if op.label is not None and len(op.failures):
        wanted = list(classes) if classes is not None else op.classes
        colors = _class_colors(wanted)
        labels = op.failures[op.label].astype(str).to_numpy()
        Xf = op.failures[sensors].to_numpy(dtype=float)
        Pf = pca.transform((Xf - mu) / sd)
        for cls in wanted:
            m = labels == cls
            if not m.any():
                continue
            sel = _subsample(int(m.sum()), max_points)
            ax.scatter(Pf[m][sel, 0], Pf[m][sel, 1], s=PLOT.scatter_size,
                       color=colors[cls], alpha=PLOT.scatter_alpha,
                       edgecolors="none", label=cls)

    var = pca.explained_variance_ratio_ * 100.0
    ax.set_xlabel("PC1 (%.1f%%)" % var[0])
    ax.set_ylabel("PC2 (%.1f%%)" % (var[1] if var.size > 1 else 0.0))
    ax.set_title("%s - PCA fitted on nominal" % op.name)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=PLOT.font_size - 2)
    fig.tight_layout()
    return fig


def trend_vs_intensity(
    op: Any,
    sensors: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[str]] = None,
    units: str = "sigma",
    signed: bool = False,
    n_bins: int = 12,
    ncols: int = 3,
    show_points: bool = True,
    max_points: Optional[int] = 1500,
):
    """Sensor response as a function of fault intensity.

    units
        ``'sigma'``    : deviation from the nominal mean in units of the nominal
                         standard deviation of that sensor - comparable across
                         sensors, and directly readable as "how many sigmas".
        ``'physical'`` : raw sensor values.

    signed
        ``False`` plots against ``|intensity|`` (the ordering that matters for
        detectability); ``True`` keeps the sign, which carries the physical
        direction of the fault.

    Each panel shows the Monte Carlo scatter, the median over quantile bins of
    intensity and the interquartile band, per class.
    """
    if op.intensity is None:
        raise ValueError(
            "operating point %r has no intensity column; set it in the manifest "
            "(columns.intensity) or pass it to infer_schema" % op.name
        )
    sensors = list(sensors) if sensors is not None else list(op.sensors)
    wanted = list(classes) if classes is not None else op.classes
    colors = _class_colors(wanted)

    Xn = op.nominal[sensors].to_numpy(dtype=float)
    mu, sd = np.nanmean(Xn, axis=0), np.nanstd(Xn, axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    fig, axes = _grid(len(sensors), ncols, width=3.4, height=2.5)
    labels = (op.failures[op.label].astype(str).to_numpy()
              if op.label is not None else np.array(["(all)"] * len(op.failures)))
    inten_all = op.intensity_values()
    if not signed:
        inten_all = np.abs(inten_all)

    for ax, name, j in zip(axes, sensors, range(len(sensors))):
        values = op.failures[name].to_numpy(dtype=float)
        y_all = (values - mu[j]) / sd[j] if units == "sigma" else values
        for cls in wanted:
            m = labels == cls
            if not m.any():
                continue
            x, y = inten_all[m], y_all[m]
            if show_points:
                sel = _subsample(x.size, max_points)
                ax.scatter(x[sel], y[sel], s=4, color=colors[cls], alpha=0.18,
                           edgecolors="none")
            stats = binned_stats(x, y, n_bins=n_bins)
            if len(stats):
                ax.plot(stats["center"], stats["median"], "-", color=colors[cls],
                        lw=1.6, label=cls)
                ax.fill_between(stats["center"], stats["q25"], stats["q75"],
                                color=colors[cls], alpha=0.18, linewidth=0)
        if units == "sigma":
            ax.axhline(0.0, color="k", lw=0.7)
            for s in (-3.0, 3.0):
                ax.axhline(s, color=PLOT.threshold_color, ls=":", lw=0.8)
            ax.set_ylabel(r"$(x-\mu)/\sigma$")
        else:
            ax.set_ylabel(name)
        ax.set_xlabel("intensity" if signed else "|intensity|")
        ax.set_title(name)

    if wanted:
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                       fontsize=PLOT.font_size - 3)
    fig.suptitle("%s - sensor trend vs fault intensity" % op.name, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# detector views
# --------------------------------------------------------------------------

def plot_score_distributions(
    scores: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
    log_x: bool = True,
    title: str = "Detector score",
    bins: int = 60,
):
    """Overlaid score histograms, e.g. ``{'nominal': ..., 'failures': ...}``."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    keys = list(scores)
    colors = dict(zip(keys, categorical_colors(len(keys))))
    for key in keys:
        v = np.asarray(scores[key], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if key.lower().startswith("nom"):
            colors[key] = PLOT.nominal_color
        ax.hist(v, bins=bins, density=True, alpha=0.55, label="%s (n=%d)" % (key, v.size),
                color=colors[key], edgecolor="none")
    if threshold is not None:
        ax.axvline(threshold, color=PLOT.threshold_color, ls="--", lw=1.3,
                   label="threshold = %.3g" % threshold)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=PLOT.font_size - 2)
    fig.tight_layout()
    return fig


def plot_contributions(
    contributions: Union[pd.Series, pd.DataFrame],
    top: int = 12,
    title: str = "Per sensor contribution to the distance",
):
    """Bar chart of the per sensor breakdown returned by a detector."""
    import matplotlib.pyplot as plt

    if isinstance(contributions, pd.DataFrame):
        series = contributions.mean(axis=0)
        title = title + " (mean over %d samples)" % len(contributions)
    else:
        series = contributions
    series = series.sort_values(ascending=False)[:top][::-1]

    fig, ax = plt.subplots(figsize=(6.0, max(2.4, 0.28 * len(series) + 1.0)))
    ax.barh(range(len(series)), series.to_numpy(dtype=float), color="#4878a8")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series.index)
    ax.set_xlabel("contribution")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pod(
    results: Any,
    show_empirical: bool = True,
    show_bounds: bool = True,
    log_x: bool = True,
    title: Optional[str] = None,
):
    """Probability of detection against fault intensity.

    ``results`` is a single ``PODResult`` or the dict returned by
    :func:`hmslib.analysis.pod_analysis`.  Empirical points carry Wilson
    intervals; the smooth line is the fitted log-odds model; the vertical marks
    are ``i90`` and, dashed, its upper confidence bound ``i90_95``.
    """
    import matplotlib.pyplot as plt

    if not isinstance(results, dict):
        results = {getattr(results, "label", "class"): results}
    keys = [k for k, r in results.items() if r.n > 0]
    colors = dict(zip(keys, categorical_colors(len(keys))))
    single = len(keys) == 1

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for key in keys:
        res = results[key]
        color = colors[key]
        if show_empirical and len(res.bins):
            b = res.bins
            yerr = np.clip(
                np.vstack([b["rate"] - b["ci_lo"], b["ci_hi"] - b["rate"]]), 0.0, None)
            ax.errorbar(b["center"], b["rate"], yerr=yerr, fmt="o", ms=3.5,
                        lw=0.0, elinewidth=0.9, capsize=2, color=color,
                        alpha=0.85 if single else 0.5)
        if res.fitted and np.isfinite(res.coef).all() and res.coef[1] > 0:
            lo = max(res.intensity_min, 1e-12)
            grid = np.geomspace(lo, res.intensity_max, 200)
            ax.plot(grid, res.pod(grid), "-", lw=1.6, color=color,
                    label="%s (i90=%.3g)" % (key, res.i90))
        else:
            ax.plot([], [], "-", lw=1.6, color=color,
                    label="%s (no fit)" % key)
        if show_bounds and np.isfinite(res.i90):
            ax.axvline(res.i90, color=color, ls=":", lw=1.0, alpha=0.8)
            if np.isfinite(res.i90_95):
                ax.axvline(res.i90_95, color=color, ls="--", lw=1.0, alpha=0.6)

    for level, style in ((0.5, ":"), (0.9, "--")):
        ax.axhline(level, color="k", ls=style, lw=0.8, alpha=0.5)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("fault intensity")
    ax.set_ylabel("probability of detection")
    alpha = next((r.alpha for r in results.values() if np.isfinite(r.alpha)), np.nan)
    ax.set_title(title or ("POD curves" + ("  (at FPR = %.3g)" % alpha
                                           if np.isfinite(alpha) else "")))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=PLOT.font_size - 2)
    fig.tight_layout()
    return fig


def plot_score_vs_intensity(
    detector: Any,
    op: Any,
    classes: Optional[Sequence[str]] = None,
    at_alpha: Optional[float] = None,
    n_bins: int = 12,
    log_y: bool = True,
    max_points: Optional[int] = 1500,
):
    """Detector score against fault intensity, with the decision threshold.

    Where the median line crosses the threshold is, visually, the detectability
    limit that :func:`hmslib.analysis.fit_pod` quantifies.
    """
    import matplotlib.pyplot as plt

    from . import analysis

    frame = analysis.score_frame(detector, op, classes, at_alpha=at_alpha)
    threshold = frame.attrs["threshold"]
    keys = list(pd.unique(frame["class"]))
    colors = _class_colors(keys)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for key in keys:
        sub = frame[frame["class"] == key]
        x = sub["intensity"].to_numpy()
        y = sub["score"].to_numpy()
        sel = _subsample(x.size, max_points)
        ax.scatter(x[sel], y[sel], s=4, color=colors[key], alpha=0.18,
                   edgecolors="none")
        stats = binned_stats(x, y, n_bins=n_bins)
        if len(stats):
            ax.plot(stats["center"], stats["median"], "-", lw=1.7,
                    color=colors[key], label=str(key))
            ax.fill_between(stats["center"], stats["q25"], stats["q75"],
                            color=colors[key], alpha=0.15, linewidth=0)
    ax.axhline(threshold, color=PLOT.threshold_color, ls="--", lw=1.4,
               label="threshold = %.3g" % threshold)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("|fault intensity|")
    ax.set_ylabel("detector score")
    ax.set_title("%s - score vs fault intensity" % op.name)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=PLOT.font_size - 2)
    fig.tight_layout()
    return fig


def plot_sensor_sensitivity(
    table: pd.DataFrame,
    metric: str = "slope",
    top: int = 8,
    ncols: int = 3,
):
    """Per class ranking of the sensors, from :func:`hmslib.analysis.sensor_sensitivity`."""
    import matplotlib.pyplot as plt

    classes = list(pd.unique(table["class"]))
    fig, axes = _grid(len(classes), ncols, width=3.4, height=2.6)
    for ax, cls in zip(axes, classes):
        sub = table[table["class"] == cls].copy()
        sub["_rank"] = sub[metric].abs()
        sub = sub.sort_values("_rank", ascending=False).head(top)[::-1]
        colors = ["#4878a8" if v >= 0 else "#a85048" for v in sub[metric]]
        ax.barh(range(len(sub)), sub[metric].to_numpy(), color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["sensor"], fontsize=PLOT.font_size - 2)
        ax.axvline(0.0, color="k", lw=0.7)
        ax.set_title(str(cls), fontsize=PLOT.font_size - 1)
        ax.set_xlabel(metric)
    fig.suptitle("Sensor sensitivity (%s, sigma per unit intensity)" % metric, y=1.0)
    fig.tight_layout()
    return fig


def plot_contribution_heatmap(
    detector: Any,
    op: Any,
    classes: Optional[Sequence[str]] = None,
    min_score: Optional[float] = None,
    title: Optional[str] = None,
):
    """Mean share of the distance carried by each sensor, per failure class.

    The isolation summary: one row per class, one column per sensor.  Only
    samples the detector actually flags contribute, since the decomposition of
    a sample sitting inside the nominal cloud says nothing useful.
    """
    import matplotlib.pyplot as plt

    wanted = list(classes) if classes is not None else op.classes
    sensors = list(detector.features_used_)
    threshold = detector.threshold_ if min_score is None else min_score

    rows, kept = [], []
    for cls in wanted:
        sub = op.failures_subset([cls])
        if not len(sub):
            continue
        X = sub[sensors]
        flagged = detector.score(X) > threshold
        if flagged.sum() < 3:
            continue
        shares = detector.contributions(X[flagged], normalize=True).mean(axis=0)
        rows.append(shares.to_numpy())
        kept.append(cls)
    if not rows:
        raise ValueError("no failure class has enough flagged samples to decompose")

    M = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(sensors) + 2.5),
                                    max(2.6, 0.30 * len(kept) + 1.6)))
    im = ax.imshow(M, aspect="auto", cmap="magma_r", interpolation="nearest",
                   vmin=0.0)
    ax.set_xticks(range(len(sensors)))
    ax.set_xticklabels(sensors, rotation=90, fontsize=PLOT.font_size - 3)
    ax.set_yticks(range(len(kept)))
    ax.set_yticklabels(kept, fontsize=PLOT.font_size - 2)
    ax.grid(False)
    for i in range(len(kept)):  # mark the dominant sensor of each class
        j = int(np.argmax(M[i]))
        ax.plot(j, i, marker="s", ms=4, mfc="none", mec="#00b0ff", mew=1.2)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean share of $d^2$")
    ax.set_title(title or "%s - distance decomposition per class" % op.name)
    fig.tight_layout()
    return fig


def plot_class_counts(op: Any, title: Optional[str] = None):
    """Horizontal bar chart of the number of runs per failure class."""
    import matplotlib.pyplot as plt

    counts = op.class_counts()
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.24 * len(counts) + 1.2)))
    if len(counts):
        ax.barh(range(len(counts)), counts.to_numpy(), color="#4878a8")
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=PLOT.font_size - 2)
    ax.set_xlabel("runs")
    ax.set_title(title or "%s - failure classes (%d runs total)" % (op.name, int(counts.sum())))
    fig.tight_layout()
    return fig
