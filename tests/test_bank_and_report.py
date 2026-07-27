"""Per-operating-point model bank, plotting and the quicklook report."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hmslib as hm
from hmslib import Mahalanobis, ModelBank, viz


# --------------------------------------------------------------------------
# ModelBank
# --------------------------------------------------------------------------

def test_bank_fits_one_model_per_operating_point(dataset):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(threshold="chi2", alpha=0.01), verbose=False)
    assert set(bank.operating_points) == set(ds.operating_points)
    for name in ds.operating_points:
        assert bank[name]._fitted
    # independently fitted: different nominal clouds, different locations
    a, b = [bank[n] for n in ds.operating_points]
    assert not np.allclose(a.scaler_.center_, b.scaler_.center_)


def test_bank_models_are_independent_copies(dataset):
    ds, _ = dataset
    template = Mahalanobis()
    bank = ModelBank.fit(ds, template, verbose=False)
    assert not template._fitted           # the template is left untouched
    names = ds.operating_points
    assert bank[names[0]] is not bank[names[1]]


def test_bank_detects_failures_at_the_right_operating_point(dataset):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(threshold="chi2", alpha=0.01), verbose=False)
    table = bank.score_dataset(ds)
    rates = table.groupby(["op", "set"])["flagged"].mean().unstack()
    assert (rates["nominal"] < 0.05).all()
    assert (rates["failure"] > rates["nominal"]).all()
    assert set(table.columns) == {"op", "set", "class", "intensity", "score", "flagged"}


def test_bank_rejects_an_unknown_operating_point(dataset):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(), verbose=False)
    with pytest.raises(KeyError):
        bank["OP_does_not_exist"]


def test_bank_with_common_sensors(dataset):
    ds, _ = dataset
    common = ds.common_sensors()
    bank = ModelBank.fit(ds, Mahalanobis(), sensors=common, verbose=False)
    for name in ds.operating_points:
        assert set(bank[name].features_in_) == set(common)


def test_bank_save_and_load(dataset, tmp_path):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(), verbose=False)
    folder = str(tmp_path / "models")
    bank.save(folder)
    assert os.path.isfile(os.path.join(folder, "bank.json"))
    back = ModelBank.load(folder)
    name = ds.operating_points[0]
    X = ds[name].nominal[bank[name].features_used_]
    assert np.allclose(back.score(X, name), bank.score(X, name))


def test_bank_to_frame(dataset):
    ds, _ = dataset
    bank = ModelBank.fit(ds, Mahalanobis(), verbose=False)
    frame = bank.to_frame()
    assert len(frame) == len(ds)
    assert {"op", "threshold", "effective_rank"} <= set(frame.columns)


# --------------------------------------------------------------------------
# viz
# --------------------------------------------------------------------------

def test_binned_stats_tracks_a_known_trend():
    x = np.linspace(0.0, 10.0, 2000)
    y = 3.0 * x + np.random.RandomState(0).normal(0, 0.1, size=x.size)
    stats = viz.binned_stats(x, y, n_bins=10)
    assert len(stats) == 10
    assert stats["median"].is_monotonic_increasing
    assert np.allclose(stats["median"], 3.0 * stats["center"], atol=0.3)
    assert (stats["q75"] >= stats["q25"]).all()


def test_binned_stats_handles_empty_and_degenerate_input():
    assert len(viz.binned_stats(np.array([]), np.array([]))) == 0
    stats = viz.binned_stats(np.ones(50), np.arange(50.0), n_bins=5)
    assert len(stats) >= 1


def test_ellipse_points_matches_the_covariance():
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    x, y = viz.ellipse_points(np.array([1.0, -2.0]), cov, radius=3.0)
    # the traced polygon touches the extremes only up to its angular resolution
    assert np.max(x) == pytest.approx(1.0 + 6.0, abs=1e-3)   # 3 sigma = 3*2
    assert np.max(y) == pytest.approx(-2.0 + 3.0, abs=1e-3)


@pytest.mark.parametrize("maker", ["hist", "corr", "spectrum", "pca", "counts", "trend"])
def test_plot_functions_produce_figures(dataset, maker):
    ds, _ = dataset
    op = ds[ds.operating_points[0]]
    if maker == "hist":
        fig = viz.plot_sensor_histograms(op.nominal, op.sensors)
    elif maker == "corr":
        fig = viz.plot_correlation_heatmap(op.nominal, op.sensors)
    elif maker == "spectrum":
        fig = viz.plot_eigen_spectrum(op.nominal, op.sensors)
    elif maker == "pca":
        fig = viz.plot_pca(op)
    elif maker == "counts":
        fig = viz.plot_class_counts(op)
    else:
        fig = viz.trend_vs_intensity(op, sensors=op.sensors[:3])
    assert fig.get_axes()
    plt.close(fig)


def test_trend_vs_intensity_requires_an_intensity_column(dataset):
    ds, _ = dataset
    op = ds[ds.operating_points[0]]
    op.schema.intensity = None
    with pytest.raises(ValueError):
        viz.trend_vs_intensity(op)
    op.schema.intensity = "Intensity"       # restore, the fixture is session scoped


def test_detector_plots(dataset):
    ds, _ = dataset
    op = ds[ds.operating_points[0]]
    det = Mahalanobis().fit(op.nominal[op.sensors])
    fig = viz.plot_score_distributions(
        {"nominal": det.score(op.nominal[op.sensors]),
         "failures": det.score(op.failures[op.sensors])},
        threshold=det.threshold_,
    )
    assert fig.get_axes()
    plt.close(fig)

    fig = viz.plot_contributions(det.contributions(op.failures[op.sensors].iloc[:20]))
    assert fig.get_axes()
    plt.close(fig)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def test_quicklook_writes_one_pdf_per_operating_point(dataset, tmp_path):
    ds, _ = dataset
    out = str(tmp_path / "reports")
    written = hm.quicklook(ds, out=out, verbose=False)
    assert len(written) == len(ds)
    for path in written:
        assert os.path.getsize(path) > 5000


def test_quicklook_single_file(dataset, tmp_path):
    ds, _ = dataset
    out = str(tmp_path / "all.pdf")
    written = hm.quicklook(ds, out=out, verbose=False)
    assert written == [out] and os.path.getsize(out) > 5000


def test_quicklook_accepts_a_bare_dataframe(op_pair, tmp_path):
    nominal, _, _ = op_pair
    out = str(tmp_path / "one.pdf")
    hm.quicklook(nominal, out=out, verbose=False)
    assert os.path.getsize(out) > 3000


def test_quicklook_accepts_a_csv_path(tmp_csv_folder, tmp_path):
    csv = os.path.join(tmp_csv_folder, "OP_60_nominal.csv")
    out = str(tmp_path / "csv.pdf")
    hm.quicklook(csv, out=out, verbose=False)
    assert os.path.getsize(out) > 3000


def test_text_page_splits_long_bodies():
    figs = hm.report.text_page("t", "\n".join("line %d" % i for i in range(200)))
    assert len(figs) >= 3
    for fig in figs:
        plt.close(fig)
