"""One-shot PDF reports.

``quicklook`` is the command to run on day one, when a new batch of data lands
on the offline machine: it reads the tables, prints what it understood of their
structure and writes a multipage PDF with everything needed to decide whether
the data are usable - shapes, missing values, constant and redundant sensors,
covariance conditioning, distributions, correlations, PCA and, when an intensity
column exists, the sensor trends against fault intensity.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import quality, viz
from .config import PLOT, apply_style

# `analysis` is imported lazily inside detection_report: it is the only module
# that pulls in scipy.stats, and quicklook must stay usable without it.

__all__ = ["quicklook", "detection_report", "text_page"]

_LINES_PER_PAGE = 62


def text_page(title: str, body: str, figsize: Any = (8.27, 11.69)) -> List[Any]:
    """Render text as one or more portrait A4 figures (monospace)."""
    import matplotlib.pyplot as plt

    lines = body.splitlines() or [""]
    pages = [lines[i:i + _LINES_PER_PAGE] for i in range(0, len(lines), _LINES_PER_PAGE)]
    figs = []
    for k, chunk in enumerate(pages):
        fig = plt.figure(figsize=figsize)
        header = title if k == 0 else "%s (cont. %d)" % (title, k + 1)
        fig.text(0.06, 0.965, header, fontsize=12, fontweight="bold", va="top")
        fig.text(0.06, 0.93, "\n".join(chunk), fontsize=7.2, va="top",
                 family="monospace", linespacing=1.35)
        figs.append(fig)
    return figs


def _op_text(op: Any, nominal_report: quality.QualityReport,
             failure_report: Optional[quality.QualityReport]) -> str:
    parts = [op.summary(), "", op.schema.summary(), "",
             "NOMINAL -- " + nominal_report.summary()]
    if failure_report is not None:
        parts += ["", "FAILURES -- " + failure_report.summary()]
    if op.intensity is not None:
        table = op.describe_intensity()
        if len(table):
            parts += ["", "INTENSITY per class:",
                      table.to_string(index=False, float_format=lambda v: "%.4g" % v)]
    counts = op.class_counts()
    if len(counts):
        parts += ["", "RUNS per class:", counts.to_string()]
    info = op.load_info or {}
    if info:
        parts += ["", "LOADING:"]
        for key, value in info.items():
            parts.append("  %-9s %s" % (
                key, {k: v for k, v in value.items() if k != "path"}))
            parts.append("            %s" % value.get("path", ""))
    return "\n".join(parts)


def _most_responsive(op: Any, k: int = 6) -> List[str]:
    """Sensors whose failure values deviate most from the nominal cloud."""
    sensors = op.sensors
    if not sensors or not len(op.failures):
        return sensors[:k]
    Xn = op.nominal[sensors].to_numpy(dtype=float)
    mu, sd = np.nanmean(Xn, axis=0), np.nanstd(Xn, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xf = op.failures[sensors].to_numpy(dtype=float)
    dev = np.nanmean(np.abs((Xf - mu) / sd), axis=0)
    order = np.argsort(-dev)
    return [sensors[i] for i in order[:k]]


def quicklook(
    target: Any,
    out: Optional[str] = None,
    max_sensors_hist: int = 60,
    trend_sensors: int = 6,
    trend_classes: int = 8,
    verbose: bool = True,
) -> List[str]:
    """Write a quick-look PDF for a dataset, an operating point or a DataFrame.

    Parameters
    ----------
    target : :class:`hmslib.io.Dataset`, :class:`hmslib.io.OperatingPointData`,
        a ``DataFrame`` or the path of a CSV file.
    out : output ``.pdf`` file, or a directory in which one PDF per operating
        point is written.  ``None`` writes ``quicklook.pdf`` in the current
        directory.
    trend_sensors : how many sensors to include in the trend-vs-intensity page,
        picked as the ones deviating most from nominal.

    Returns the list of files written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    apply_style()
    ops = _as_operating_points(target)
    out = out or "quicklook.pdf"
    is_dir = out.endswith(("/", "\\")) or (not out.lower().endswith(".pdf"))
    written: List[str] = []

    if is_dir:
        os.makedirs(out, exist_ok=True)
        for op in ops:
            path = os.path.join(out, "quicklook_%s.pdf" % _safe(op.name))
            _write_pdf([op], path, max_sensors_hist, trend_sensors, trend_classes, verbose)
            written.append(path)
    else:
        folder = os.path.dirname(os.path.abspath(out))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        _write_pdf(ops, out, max_sensors_hist, trend_sensors, trend_classes, verbose)
        written.append(out)

    if verbose:
        for path in written:
            print("quicklook written to %s" % path)
    return written


def _write_pdf(
    ops: Sequence[Any], path: str, max_sensors_hist: int,
    trend_sensors: int, trend_classes: int, verbose: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for op in ops:
            if verbose:
                print("quicklook: %s" % op.name)
            sensors = op.sensors
            rep_nom = quality.check(op.nominal, sensors)
            rep_fail = quality.check(op.failures, sensors) if len(op.failures) else None

            for fig in text_page("Quicklook - %s" % op.name,
                                 _op_text(op, rep_nom, rep_fail)):
                pdf.savefig(fig)
                plt.close(fig)

            shown = sensors[:max_sensors_hist]
            if shown:
                fig = viz.plot_sensor_histograms(
                    op.nominal, shown, standardize=True,
                    title="%s - nominal distributions (standardized)" % op.name)
                pdf.savefig(fig)
                plt.close(fig)

                fig = viz.plot_correlation_heatmap(
                    op.nominal, shown, title="%s - nominal correlation" % op.name)
                pdf.savefig(fig)
                plt.close(fig)

                fig = viz.plot_eigen_spectrum(
                    op.nominal, shown,
                    title="%s - nominal correlation spectrum" % op.name)
                pdf.savefig(fig)
                plt.close(fig)

            if len(op.failures):
                classes = op.classes[:trend_classes]
                try:
                    fig = viz.plot_pca(op, classes=classes)
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as exc:  # keep the report going
                    _note(pdf, "PCA page failed: %s" % exc)

                fig = viz.plot_class_counts(op)
                pdf.savefig(fig)
                plt.close(fig)

                if op.intensity is not None:
                    try:
                        fig = viz.trend_vs_intensity(
                            op, sensors=_most_responsive(op, trend_sensors),
                            classes=classes)
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception as exc:
                        _note(pdf, "trend page failed: %s" % exc)


def detection_report(
    bank: Any,
    dataset: Any,
    out: str = "detection_report.pdf",
    at_alpha: Optional[float] = None,
    classes: Optional[Sequence[str]] = None,
    n_boot: int = 200,
    max_classes_plot: int = 10,
    verbose: bool = True,
) -> str:
    """Detection and detectability report for a fitted :class:`hmslib.ModelBank`.

    One section per operating point: score separation, score against fault
    intensity, POD curves with ``i50 / i90 / i90_95``, the sensor sensitivity
    ranking and the per class decomposition of the distance.

    ``at_alpha`` re-calibrates every detector on its own nominal cloud before
    the analysis, so all the numbers hold at one stated false positive rate -
    the only way POD figures are comparable.  Without it, each detector's own
    calibrated threshold is used.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from . import analysis

    apply_style()
    folder = os.path.dirname(os.path.abspath(out))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    with PdfPages(out) as pdf:
        for name, op in dataset.items():
            if name not in bank:
                continue
            if verbose:
                print("detection report: %s" % name)
            det = bank[name]
            sensors = det.features_used_
            wanted = list(classes) if classes is not None else op.classes

            if at_alpha is not None:
                det.calibrate_threshold(op.nominal[sensors], alpha=at_alpha)

            s_nom = det.score(op.nominal[sensors])
            s_fail = det.score(op.failures[sensors]) if len(op.failures) else np.array([])

            header = [
                "operating point : %s" % name,
                "detector        : %s" % type(det).__name__,
                "threshold       : %.6g (%s)"
                % (det.threshold_, det.diagnostics_.get("threshold_rule", "?")),
                "sensors used    : %d of %d" % (len(sensors), len(op.sensors)),
                "",
                "false positives on nominal : %.3f%% (%d of %d)"
                % (100 * np.mean(s_nom > det.threshold_),
                   int(np.sum(s_nom > det.threshold_)), s_nom.size),
            ]
            if s_fail.size:
                header.append("detected failures          : %.1f%% (%d of %d)"
                              % (100 * np.mean(s_fail > det.threshold_),
                                 int(np.sum(s_fail > det.threshold_)), s_fail.size))
            if det.warnings_:
                header += ["", "detector warnings:"] + ["  ! " + w for w in det.warnings_]

            results: Dict[str, Any] = {}
            if op.intensity is not None and len(op.failures):
                try:
                    results = analysis.pod_analysis(
                        det, op, classes=wanted, n_boot=n_boot)
                    table = analysis.pod_table(results)
                    header += [
                        "", "DETECTABILITY (hardest classes first)",
                        table.to_string(index=False,
                                        float_format=lambda v: "%.4g" % v),
                        "",
                        "i50/i90   : intensity detected 50% / 90% of the time",
                        "i90_95    : upper 95% confidence bound on i90",
                        "_emp      : same, read off the binned empirical curve",
                    ]
                    notes = [(c, n) for c, r in results.items() for n in r.notes]
                    if notes:
                        header += ["", "notes:"] + ["  [%s] %s" % (c, n) for c, n in notes]
                except Exception as exc:
                    header += ["", "POD analysis failed: %s" % exc]
            else:
                header += ["", "no intensity column: POD analysis skipped"]

            for fig in text_page("Detection - %s" % name, "\n".join(header)):
                pdf.savefig(fig)
                plt.close(fig)

            plots = [
                ("score separation", lambda: viz.plot_score_distributions(
                    {"nominal": s_nom, "failures": s_fail}, threshold=det.threshold_,
                    title="%s - score separation" % name)),
            ]
            if op.intensity is not None and len(op.failures):
                sel = wanted[:max_classes_plot]
                plots += [
                    ("score vs intensity", lambda: viz.plot_score_vs_intensity(
                        det, op, classes=sel)),
                    ("POD curves", lambda: viz.plot_pod(
                        {k: v for k, v in results.items() if k in sel})),
                    ("sensor sensitivity", lambda: viz.plot_sensor_sensitivity(
                        analysis.sensor_sensitivity(op, classes=sel[:6]))),
                ]
            if len(op.failures):
                plots.append(("distance decomposition",
                              lambda: viz.plot_contribution_heatmap(
                                  det, op, classes=wanted[:20])))

            for label, maker in plots:
                try:
                    fig = maker()
                except Exception as exc:
                    _note(pdf, "page %r failed: %s" % (label, exc))
                    continue
                pdf.savefig(fig)
                plt.close(fig)

    if verbose:
        print("detection report written to %s" % out)
    return out


def _note(pdf: Any, message: str) -> None:
    import matplotlib.pyplot as plt

    for fig in text_page("Note", message):
        pdf.savefig(fig)
        plt.close(fig)


def _as_operating_points(target: Any) -> List[Any]:
    from .io import Dataset, OperatingPointData, load_csv
    from .schema import infer_schema

    if isinstance(target, Dataset):
        return list(target)
    if isinstance(target, OperatingPointData):
        return [target]
    if isinstance(target, str):
        df, _info = load_csv(target)
        name = os.path.splitext(os.path.basename(target))[0]
        return _as_operating_points((df, name))
    if isinstance(target, tuple) and len(target) == 2:
        df, name = target
    elif isinstance(target, pd.DataFrame):
        df, name = target, "table"
    else:
        raise TypeError(
            "quicklook accepts a Dataset, an OperatingPointData, a DataFrame or a "
            "CSV path, got %s" % type(target).__name__
        )
    sch = infer_schema(df)
    empty = df.iloc[0:0]
    if sch.label is None:
        # no label column: treat the whole table as a nominal cloud
        return [OperatingPointData(name, df, empty, sch)]
    return [OperatingPointData(name, df, df, sch)]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
