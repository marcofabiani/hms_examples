"""Detectability and POD curves, checked against known ground truth."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hmslib as hm
from hmslib import Mahalanobis, ModelBank, analysis, viz


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def test_wilson_interval_brackets_the_proportion():
    lo, hi = analysis.wilson_interval(30, 100)
    assert lo < 0.30 < hi
    assert 0.0 <= lo and hi <= 1.0
    # at the boundary the interval stays inside [0, 1], unlike the normal one
    lo, hi = analysis.wilson_interval(100, 100)
    assert hi == 1.0 and 0.9 < lo < 1.0
    lo, hi = analysis.wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.1
    # wider with less data
    narrow = analysis.wilson_interval(500, 1000)
    wide = analysis.wilson_interval(5, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_irls_recovers_known_logistic_coefficients():
    rng = np.random.RandomState(0)
    b0, b1 = -6.9, 3.0
    x = rng.uniform(np.log(1.0), np.log(100.0), size=40000)
    p = 1.0 / (1.0 + np.exp(-(b0 + b1 * x)))
    y = (rng.uniform(size=x.size) < p).astype(float)
    beta, converged = analysis._logistic_irls(x, y)
    assert converged
    assert beta[0] == pytest.approx(b0, rel=0.05)
    assert beta[1] == pytest.approx(b1, rel=0.05)


def test_empirical_pod_is_monotone_on_monotone_data():
    rng = np.random.RandomState(1)
    x = np.exp(rng.uniform(np.log(1.0), np.log(100.0), size=5000))
    p = 1.0 / (1.0 + np.exp(-(-6.9 + 3.0 * np.log(x))))
    y = rng.uniform(size=x.size) < p
    bins = analysis.empirical_pod(x, y, n_bins=10)
    assert len(bins) == 10
    assert bins["n"].sum() == x.size
    assert bins["rate"].is_monotonic_increasing
    assert ((bins["ci_lo"] <= bins["rate"]) & (bins["rate"] <= bins["ci_hi"])).all()


# --------------------------------------------------------------------------
# POD fitting
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def known_pod():
    """Detection generated from a POD curve with i50 = 10 exactly."""
    rng = np.random.RandomState(2)
    b1 = 3.0
    b0 = -b1 * np.log(10.0)             # i50 = 10 by construction
    x = np.exp(rng.uniform(np.log(1.0), np.log(200.0), size=20000))
    p = 1.0 / (1.0 + np.exp(-(b0 + b1 * np.log(x))))
    y = rng.uniform(size=x.size) < p
    i90_true = float(np.exp((np.log(9.0) - b0) / b1))
    return x, y, 10.0, i90_true


def test_fit_pod_recovers_i50_and_i90(known_pod):
    x, y, i50_true, i90_true = known_pod
    res = analysis.fit_pod(x, y, label="known", n_boot=0)
    assert res.fitted and not res.separated
    assert res.i50 == pytest.approx(i50_true, rel=0.06)
    assert res.i90 == pytest.approx(i90_true, rel=0.08)
    assert res.i90 > res.i50
    assert res.pod(res.i50) == pytest.approx(0.5, abs=0.01)
    assert res.pod(res.i90) == pytest.approx(0.9, abs=0.01)


def test_empirical_and_parametric_estimates_agree(known_pod):
    x, y, _, _ = known_pod
    res = analysis.fit_pod(x, y, n_bins=14, n_boot=0)
    assert res.i50_empirical == pytest.approx(res.i50, rel=0.35)
    assert res.i90_empirical == pytest.approx(res.i90, rel=0.35)


def test_confidence_bound_is_conservative(known_pod):
    x, y, _, _ = known_pod
    res = analysis.fit_pod(x[:3000], y[:3000], n_boot=150, random_state=0)
    assert np.isfinite(res.i90_95)
    assert res.i90_95 >= res.i90
    assert res.boot_i90.size >= 100


def test_confidence_bound_tightens_with_more_data(known_pod):
    x, y, _, _ = known_pod
    small = analysis.fit_pod(x[:800], y[:800], n_boot=150, random_state=0)
    large = analysis.fit_pod(x, y, n_boot=150, random_state=0)
    assert (small.i90_95 / small.i90) > (large.i90_95 / large.i90)


def test_always_detected_is_flagged_not_fitted_silently():
    x = np.exp(np.linspace(np.log(5.0), np.log(50.0), 500))
    res = analysis.fit_pod(x, np.ones(500, dtype=bool), label="easy")
    assert res.separated
    assert res.detection_rate == 1.0
    assert np.isnan(res.i90)
    assert any("always detected" in n for n in res.notes)


def test_never_detected_is_flagged():
    x = np.exp(np.linspace(np.log(5.0), np.log(50.0), 500))
    res = analysis.fit_pod(x, np.zeros(500, dtype=bool), label="invisible")
    assert res.detection_rate == 0.0
    assert np.isnan(res.i90)
    assert any("never detected" in n for n in res.notes)


def test_zero_and_missing_intensities_are_excluded():
    rng = np.random.RandomState(3)
    x = np.concatenate([np.exp(rng.uniform(0, 4, 900)), np.zeros(50), np.full(50, np.nan)])
    y = rng.uniform(size=x.size) < 0.5
    res = analysis.fit_pod(x, y)
    assert res.n == 900
    assert any("zero or missing intensity" in n for n in res.notes)


def test_extrapolated_i90_is_flagged():
    """A sweep that stops before the fault becomes visible must say so."""
    rng = np.random.RandomState(4)
    x = np.exp(rng.uniform(np.log(1.0), np.log(5.0), size=4000))
    p = 1.0 / (1.0 + np.exp(-(-3.0 * np.log(10.0) + 3.0 * np.log(x))))
    y = rng.uniform(size=x.size) < p
    res = analysis.fit_pod(x, y, n_boot=0)
    assert res.i90 > res.intensity_max
    assert any("beyond the sampled range" in n for n in res.notes)


def test_fit_pod_with_no_usable_data():
    res = analysis.fit_pod(np.array([]), np.array([], dtype=bool))
    assert res.n == 0 and not res.fitted
    assert np.isnan(res.pod(5.0)).all()


# --------------------------------------------------------------------------
# end to end on synthetic engines
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_op():
    # no redundant sensors here: the duplicate is a verbatim copy of a possibly
    # affected sensor, which would make the "affected sensors" ground truth
    # ambiguous for the sensitivity tests
    nominal, failures, truth = hm.synth.make_operating_point(
        n_sensors=10, n_nominal=2000, n_classes=3, n_per_class=800,
        effect_scale=0.05, intensity_max=60.0, random_state=21,
        duplicate_sensor=False, collinear_sensor=False,
    )
    ds = hm.io.Dataset.from_frames(nominal, failures, name="OP_test")
    op = ds["OP_test"]
    det = Mahalanobis(threshold="chi2", alpha=1e-3).fit(op.nominal[op.sensors])
    return op, det, truth


def test_score_frame_shape_and_threshold(fitted_op):
    op, det, _ = fitted_op
    frame = analysis.score_frame(det, op)
    assert len(frame) == len(op.failures)
    assert set(frame.columns) == {"class", "intensity", "intensity_signed",
                                  "score", "detected"}
    assert (frame["intensity"] >= 0).all()
    assert frame.attrs["threshold"] == det.threshold_
    assert np.array_equal(frame["detected"], frame["score"] > det.threshold_)


def test_score_frame_at_alpha_leaves_the_detector_untouched(fitted_op):
    op, det, _ = fitted_op
    before = det.threshold_
    frame = analysis.score_frame(det, op, at_alpha=0.05)
    assert det.threshold_ == before
    assert frame.attrs["threshold"] != before


def test_looser_false_positive_rate_lowers_i90(fitted_op):
    op, det, _ = fitted_op
    strict = analysis.pod_analysis(det, op, at_alpha=1e-4, n_boot=0)
    loose = analysis.pod_analysis(det, op, at_alpha=5e-2, n_boot=0)
    compared = 0
    for cls in strict:
        a, b = strict[cls].i90, loose[cls].i90
        if np.isfinite(a) and np.isfinite(b):
            assert b <= a * 1.02, (cls, a, b)
            compared += 1
    assert compared >= 1


def test_stronger_faults_are_detected_at_lower_intensity():
    """The sanity check that makes i90 meaningful as a physical number."""
    def median_i90(effect_scale):
        nominal, failures, _ = hm.synth.make_operating_point(
            n_sensors=10, n_nominal=2000, n_classes=3, n_per_class=800,
            effect_scale=effect_scale, intensity_max=60.0, random_state=33,
        )
        ds = hm.io.Dataset.from_frames(nominal, failures, name="OP")
        op = ds["OP"]
        det = Mahalanobis(threshold="chi2", alpha=1e-3).fit(op.nominal[op.sensors])
        results = analysis.pod_analysis(det, op, n_boot=0)
        values = [r.i90 for r in results.values() if np.isfinite(r.i90)]
        return float(np.median(values)) if values else np.inf

    assert median_i90(0.10) < median_i90(0.04)


def test_pod_matches_the_analytical_curve(fitted_op):
    """Validate the whole chain against theory.

    For a Gaussian nominal cloud and a fault that shifts the mean by
    ``delta``, the squared Mahalanobis distance follows a non central
    chi-square with ``p`` degrees of freedom and non centrality
    ``lambda = delta' Sigma^-1 delta``.  The true POD at intensity ``i`` is
    therefore ``P(chi2'(p, lambda(i)) > threshold^2)``, computable in closed
    form - and the fitted log-odds curve must land on it.
    """
    from scipy import optimize, stats as sps

    op, det, truth = fitted_op
    sensors = list(det.features_used_)
    order = [truth["sensors"].index(s) for s in sensors]
    precision = det.precision_
    p = det.n_components_
    thr2 = det.threshold_ ** 2
    scale = truth["effect_scale"]

    results = analysis.pod_analysis(det, op, n_boot=0)
    checked = 0
    for cls, res in results.items():
        if not np.isfinite(res.i90):
            continue
        d = truth["directions"][cls][order]
        quad = float(d @ precision @ d)          # lambda = (i * scale)^2 * quad

        def pod_true(i):
            return float(sps.ncx2.sf(thr2, p, (i * scale) ** 2 * quad))

        # the analytical curve, read at the estimated i90, must be near 0.9
        assert pod_true(res.i90) == pytest.approx(0.9, abs=0.06), (cls, res.i90)

        i90_true = optimize.brentq(lambda i: pod_true(i) - 0.9, 1e-6, 1e6)
        assert res.i90 == pytest.approx(i90_true, rel=0.15), (cls, res.i90, i90_true)
        checked += 1
    assert checked >= 2


def test_pod_table_layout(fitted_op):
    op, det, _ = fitted_op
    results = analysis.pod_analysis(det, op, n_boot=50)
    table = analysis.pod_table(results)
    assert len(table) == len(op.classes)
    assert {"class", "n", "i50", "i90", "i90_95", "detection_rate"} <= set(table.columns)
    finite = table["i90"].dropna()
    assert finite.is_monotonic_decreasing        # hardest classes first
    assert table["n"].sum() == len(op.failures)


def test_pod_results_carry_the_operating_conditions(fitted_op):
    op, det, _ = fitted_op
    results = analysis.pod_analysis(det, op, at_alpha=1e-3, n_boot=0)
    for res in results.values():
        assert res.alpha == 1e-3
        assert np.isfinite(res.threshold)
        assert res.summary().startswith("POD - ")


def test_pod_analysis_needs_an_intensity_column(dataset):
    ds, _ = dataset
    op = ds[ds.operating_points[0]]
    det = Mahalanobis().fit(op.nominal[op.sensors])
    op.schema.intensity = None
    try:
        with pytest.raises(ValueError):
            analysis.pod_analysis(det, op)
    finally:
        op.schema.intensity = "Intensity"


def test_detectability_summary_covers_every_operating_point(dataset):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(threshold="chi2", alpha=1e-3), verbose=False)
    table = analysis.detectability_summary(bank, ds, n_boot=0)
    assert set(table["op"]) == set(ds.operating_points)
    assert len(table) == sum(len(ds[n].classes) for n in ds.operating_points)


# --------------------------------------------------------------------------
# sensor level
# --------------------------------------------------------------------------

def test_sensor_sensitivity_ranks_the_affected_sensors_first(fitted_op):
    op, _, truth = fitted_op
    table = analysis.sensor_sensitivity(op)
    assert len(table) == len(op.classes) * len(op.sensors)
    for cls, affected in truth["affected"].items():
        if cls not in set(table["class"]):
            continue
        top = list(table[table["class"] == cls]["sensor"].head(len(affected)))
        assert set(top) == set(affected), (cls, top, affected)


def test_sensor_sensitivity_slope_sign_follows_the_fault_direction(fitted_op):
    op, _, truth = fitted_op
    table = analysis.sensor_sensitivity(op).set_index(["class", "sensor"])
    names = truth["sensors"]
    for cls, direction in truth["directions"].items():
        if cls not in {c for c, _ in table.index}:
            continue
        for j, name in enumerate(names):
            if abs(direction[j]) < 0.3:
                continue
            assert np.sign(table.loc[(cls, name), "slope"]) == np.sign(direction[j])


def test_sensor_sensitivity_correlation_is_higher_for_affected_sensors(fitted_op):
    op, _, truth = fitted_op
    table = analysis.sensor_sensitivity(op)
    for cls, affected in truth["affected"].items():
        sub = table[table["class"] == cls]
        hit = sub[sub["sensor"].isin(affected)]["r"].abs().mean()
        miss = sub[~sub["sensor"].isin(affected)]["r"].abs().mean()
        assert hit > miss


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def test_plot_pod_accepts_dict_and_single_result(fitted_op):
    op, det, _ = fitted_op
    results = analysis.pod_analysis(det, op, n_boot=30)
    fig = viz.plot_pod(results)
    assert fig.get_axes()
    plt.close(fig)
    fig = viz.plot_pod(list(results.values())[0])
    assert fig.get_axes()
    plt.close(fig)


def test_plot_score_vs_intensity(fitted_op):
    op, det, _ = fitted_op
    fig = viz.plot_score_vs_intensity(det, op)
    assert fig.get_axes()
    plt.close(fig)


def test_plot_sensor_sensitivity(fitted_op):
    op, _, _ = fitted_op
    fig = viz.plot_sensor_sensitivity(analysis.sensor_sensitivity(op))
    assert fig.get_axes()
    plt.close(fig)


def test_plot_contribution_heatmap(fitted_op):
    op, det, _ = fitted_op
    fig = viz.plot_contribution_heatmap(det, op)
    ax = fig.get_axes()[0]
    assert len(ax.get_yticklabels()) <= len(op.classes)
    plt.close(fig)


# --------------------------------------------------------------------------
# detection report
# --------------------------------------------------------------------------

def test_detection_report_is_written(dataset, tmp_path):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(threshold="chi2", alpha=1e-3), verbose=False)
    out = str(tmp_path / "detection.pdf")
    hm.detection_report(bank, ds, out=out, n_boot=30, verbose=False)
    import os
    assert os.path.getsize(out) > 20000


def test_detection_report_at_alpha_recalibrates(dataset, tmp_path):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(threshold="chi2", alpha=1e-3), verbose=False)
    before = bank[ds.operating_points[0]].threshold_
    hm.detection_report(bank, ds, out=str(tmp_path / "d.pdf"), at_alpha=0.05,
                        n_boot=0, verbose=False)
    assert bank[ds.operating_points[0]].threshold_ < before


def test_detection_report_survives_a_missing_intensity_column(dataset, tmp_path):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(), verbose=False)
    saved = {n: ds[n].schema.intensity for n in ds.operating_points}
    for n in ds.operating_points:
        ds[n].schema.intensity = None
    try:
        out = str(tmp_path / "no_intensity.pdf")
        hm.detection_report(bank, ds, out=out, n_boot=0, verbose=False)
        import os
        assert os.path.getsize(out) > 10000
    finally:
        for n, value in saved.items():
            ds[n].schema.intensity = value
