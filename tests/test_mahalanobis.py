"""The Mahalanobis chain, checked against properties known analytically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import hmslib as hm
from hmslib import Mahalanobis


# --------------------------------------------------------------------------
# statistical correctness on clean, well conditioned data
# --------------------------------------------------------------------------

def test_squared_distance_follows_chi2(gaussian_clean):
    """With a correct covariance, E[d^2] = p and the distribution is chi2_p."""
    train, test, _ = gaussian_clean
    p = train.shape[1]
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                      threshold="chi2", calib_fraction=0.0)
    det.fit(train)
    d2 = det.squared_distance(test)
    assert d2.mean() == pytest.approx(p, rel=0.05)
    # Kolmogorov-Smirnov against chi2 with p degrees of freedom
    stat = stats.kstest(d2, "chi2", args=(p,)).statistic
    assert stat < 0.02


def test_chi2_threshold_delivers_the_requested_false_positive_rate(gaussian_clean):
    train, test, _ = gaussian_clean
    for alpha in (0.05, 0.01, 0.001):
        det = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                          threshold="chi2", alpha=alpha, calib_fraction=0.0)
        det.fit(train)
        fpr = det.false_positive_rate(test)
        assert 0.4 * alpha < fpr < 2.5 * alpha, (alpha, fpr)


def test_empirical_threshold_is_calibrated_out_of_sample(gaussian_clean):
    train, test, _ = gaussian_clean
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                      threshold="empirical", alpha=0.01, calib_fraction=0.3)
    det.fit(train)
    assert det.diagnostics_["calibration_samples"] == pytest.approx(6000, rel=0.02)
    assert det.false_positive_rate(test) == pytest.approx(0.01, abs=0.004)


def test_thresholds_are_all_available_and_close_on_clean_data(gaussian_clean):
    train, _, _ = gaussian_clean
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False, alpha=0.01)
    det.fit(train)
    assert set(det.thresholds_) == {"chi2", "hotelling", "empirical"}
    values = np.array(list(det.thresholds_.values()))
    assert np.ptp(values) / values.mean() < 0.1
    assert det.threshold_table()["selected"].sum() == 1


def test_in_sample_threshold_is_optimistic(gaussian_clean):
    """The reason calib_fraction defaults to a held-out split."""
    train, test, _ = gaussian_clean
    small = train[:400]
    out_of_sample = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                                threshold="empirical", alpha=0.05,
                                calib_fraction=0.3, random_state=0).fit(small)
    in_sample = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                            threshold="empirical", alpha=0.05,
                            calib_fraction=0.0, random_state=0).fit(small)
    in_sample.calibrate_threshold(small, alpha=0.05)
    # the in-sample rule under-estimates the tail, so it fires more often on
    # fresh data than the requested 5%
    assert in_sample.false_positive_rate(test) > out_of_sample.false_positive_rate(test)


# --------------------------------------------------------------------------
# robustness to rank deficiency
# --------------------------------------------------------------------------

def test_duplicated_sensor_is_dropped_by_default(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    det = Mahalanobis().fit(nominal[sensors])
    assert set(det.features_used_) == set(truth["sensors"])
    assert det.dropped_sensors_[truth["sensors"][0] + "_bis"] == truth["sensors"][0]
    assert np.isfinite(det.diagnostics_["condition_number"])


def test_singular_covariance_does_not_crash_cholesky(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                      inverse="cholesky").fit(nominal[sensors])
    s = det.score(nominal[sensors])
    assert np.all(np.isfinite(s))
    assert det.diagnostics_["effective_rank"] < len(sensors)


def test_eigen_inverse_truncates_to_the_effective_rank(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                      inverse="eigen").fit(nominal[sensors])
    assert det.n_components_ == det.diagnostics_["effective_rank"]
    assert det.n_components_ < len(sensors)
    assert any("truncated" in w for w in det.warnings_)


def test_var_explained_controls_the_number_of_components(gaussian_clean):
    train, _, _ = gaussian_clean
    loose = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                        inverse="eigen", var_explained=0.80).fit(train)
    tight = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                        inverse="eigen", var_explained=0.999).fit(train)
    assert loose.n_components_ < tight.n_components_ <= train.shape[1]


def test_dropping_redundant_sensors_keeps_the_false_positive_rate_honest(op_pair):
    """The headline reason for the pre-check.

    Keeping a duplicated sensor leaves a direction with no nominal variance;
    the chi-square threshold then assumes more degrees of freedom than the data
    actually carry and the false positive rate drifts away from the target.
    """
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    train, test = nominal.iloc[:1000], nominal.iloc[1000:]
    alpha = 0.01

    clean = Mahalanobis(cov="empirical", drop_redundant_sensors=True,
                        threshold="chi2", alpha=alpha).fit(train[sensors])
    naive = Mahalanobis(cov="empirical", drop_redundant_sensors=False,
                        threshold="chi2", alpha=alpha).fit(train[sensors])
    err_clean = abs(clean.false_positive_rate(test[sensors]) - alpha)
    err_naive = abs(naive.false_positive_rate(test[sensors]) - alpha)
    assert err_clean <= err_naive


def test_warns_when_samples_per_sensor_is_small(gaussian_clean):
    train, _, _ = gaussian_clean
    det = Mahalanobis(cov="ledoit_wolf", drop_redundant_sensors=False).fit(train[:40])
    assert any("samples per sensor" in w for w in det.warnings_)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cov", ["empirical", "ledoit_wolf", "oas", "mcd", "diagonal"])
def test_every_covariance_estimator_runs(cov, op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis(cov=cov, scaler="robust" if cov == "mcd" else "standard")
    det.fit(nominal[truth["sensors"]])
    s_nom = det.score(nominal[truth["sensors"]])
    s_fail = det.score(failures[truth["sensors"]])
    assert np.all(np.isfinite(s_nom)) and np.all(np.isfinite(s_fail))
    assert s_fail.mean() > s_nom.mean()


def test_mcd_resists_contamination_of_the_nominal_set(op_pair):
    """A few gross outliers in the nominal set inflate the classic covariance.

    The inflated covariance shrinks every distance, so the detector becomes
    blind.  Measured on the distances themselves rather than on a rate, which
    would be dominated by sampling noise at small alpha.
    """
    nominal, _, truth = op_pair
    sensors = truth["sensors"]
    clean = nominal[sensors].iloc[:1000].copy()
    fresh = nominal[sensors].iloc[1000:]

    # 5% of the runs come from a much wider regime: the contamination inflates
    # the covariance in every direction, not just one
    dirty = clean.copy()
    center = clean.mean().to_numpy()
    dirty.iloc[:50] = center + (clean.iloc[:50].to_numpy() - center) * 10.0

    def mean_distance(cov, scaler, data):
        det = Mahalanobis(cov=cov, scaler=scaler, calib_fraction=0.0,
                          drop_redundant_sensors=False).fit(data)
        return float(det.score(fresh).mean())

    classic = (mean_distance("empirical", "standard", dirty)
               / mean_distance("empirical", "standard", clean))
    robust = (mean_distance("mcd", "robust", dirty)
              / mean_distance("mcd", "robust", clean))
    assert classic < 0.8, classic     # contamination visibly deflates distances
    assert 0.9 < robust < 1.1, robust  # MCD barely notices


# --------------------------------------------------------------------------
# contributions
# --------------------------------------------------------------------------

def test_contributions_sum_to_the_squared_distance(op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    sample = failures[truth["sensors"]].iloc[:200]
    contrib = det.contributions(sample)
    assert np.allclose(contrib.sum(axis=1).to_numpy(),
                       det.squared_distance(sample), rtol=1e-8)
    assert list(contrib.columns) == det.features_used_


def test_normalized_contributions_sum_to_one(op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    shares = det.contributions(failures[truth["sensors"]].iloc[:100], normalize=True)
    assert np.allclose(shares.sum(axis=1).to_numpy(), 1.0)


def test_contributions_point_at_the_affected_sensors(op_pair):
    """Isolation, not just detection: the decomposition must name the culprit."""
    nominal, failures, truth = op_pair
    sensors = truth["sensors"]
    det = Mahalanobis().fit(nominal[sensors])

    for cls, affected in truth["affected"].items():
        sub = failures[failures["Failure"] == cls]
        strong = sub.nlargest(60, "Intensity")          # unambiguous cases only
        shares = det.contributions(strong[sensors], normalize=True).mean(axis=0)
        top = list(shares.sort_values(ascending=False).index[:len(affected)])
        assert set(top) & set(affected), (cls, top, affected)


def test_top_contributors_shape(op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    table = det.top_contributors(failures[truth["sensors"]].iloc[:5], k=3)
    assert len(table) == 5
    assert "sensor_1" in table.columns and "share_3" in table.columns


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def test_scoring_matches_columns_by_name(op_pair):
    nominal, failures, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    det = Mahalanobis().fit(nominal[sensors])
    reference = det.score(failures[sensors])
    shuffled = failures[list(reversed(sensors))]
    assert np.allclose(det.score(shuffled), reference)
    with_extras = failures.copy()
    assert np.allclose(det.score(with_extras), reference)


def test_scoring_rejects_a_missing_sensor(op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    with pytest.raises(KeyError):
        det.score(failures[truth["sensors"][1:]])


def test_predict_is_score_above_threshold(op_pair):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    X = failures[truth["sensors"]]
    assert np.array_equal(det.predict(X), det.score(X) > det.threshold_)


def test_use_threshold_switches_rule(gaussian_clean):
    train, _, _ = gaussian_clean
    det = Mahalanobis(cov="empirical", drop_redundant_sensors=False).fit(train)
    det.use_threshold("chi2")
    assert det.threshold_ == det.thresholds_["chi2"]
    assert det.diagnostics_["threshold_rule"] == "chi2"
    with pytest.raises(KeyError):
        det.use_threshold("nonsense")


def test_save_and_load_roundtrip(op_pair, tmp_path):
    nominal, failures, truth = op_pair
    det = Mahalanobis().fit(nominal[truth["sensors"]])
    path = str(tmp_path / "model.joblib")
    det.save(path)
    assert (tmp_path / "model.json").is_file()
    back = hm.BaseDetector.load(path)
    X = failures[truth["sensors"]]
    assert np.allclose(back.score(X), det.score(X))
    assert back.threshold_ == det.threshold_
    assert back.features_used_ == det.features_used_


def test_unfitted_model_refuses_to_score(op_pair):
    nominal, _, truth = op_pair
    det = Mahalanobis()
    with pytest.raises(RuntimeError):
        det.score(nominal[truth["sensors"]])


@pytest.mark.parametrize("kwargs", [
    {"cov": "nope"}, {"inverse": "nope"}, {"threshold": "nope"},
    {"alpha": 0.0}, {"alpha": 1.0}, {"calib_fraction": 1.0},
])
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Mahalanobis(**kwargs)


def test_fit_refuses_an_all_redundant_input():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    with pytest.raises(ValueError):
        Mahalanobis().fit(df)
