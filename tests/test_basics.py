"""compat, schema, quality and preprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hmslib as hm
from hmslib import preprocess, quality, schema


# --------------------------------------------------------------------------
# compat
# --------------------------------------------------------------------------

def test_version_tuple_parses_messy_versions():
    assert hm.compat.version_tuple("2.3.5") == (2, 3, 5)
    assert hm.compat.version_tuple("1.26.4.post1") == (1, 26, 4)
    assert hm.compat.version_tuple("2.9.1+cpu") == (2, 9, 1)
    assert hm.compat.version_tuple("3") == (3, 0, 0)


def test_check_env_reports_required_packages():
    info = hm.check_env(verbose=False)
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib"):
        assert info[name] is not None


def test_categorical_colors_returns_requested_count():
    assert len(hm.compat.categorical_colors(37)) == 37
    assert len(hm.compat.categorical_colors(3)) == 3


def test_trapezoid_wrapper():
    x = np.linspace(0.0, 1.0, 101)
    assert hm.compat.trapezoid(x, x) == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_infer_schema_finds_label_and_intensity(op_pair):
    nominal, failures, truth = op_pair
    sch = schema.infer_schema(failures, nominal)
    assert sch.label == "Failure"
    assert sch.intensity == "Intensity"
    expected = set(truth["sensors"]) | set(truth["extra_sensors"])
    assert set(sch.sensors) == expected
    assert "Failure" not in sch.sensors and "Intensity" not in sch.sensors


def test_infer_schema_respects_explicit_overrides(op_pair):
    nominal, failures, _ = op_pair
    sch = schema.infer_schema(failures, nominal, label="Failure", intensity="none")
    assert sch.intensity is None
    # it is no longer claimed as intensity, but it cannot become a sensor
    # either: it does not exist in the nominal table
    assert "Intensity" not in sch.sensors
    assert "Intensity" in sch.excluded


def test_infer_schema_excludes_constant_columns(op_pair):
    nominal, failures, _ = op_pair
    nominal = nominal.copy()
    failures = failures.copy()
    nominal["Dead"] = 1.0
    failures["Dead"] = 1.0
    sch = schema.infer_schema(failures, nominal)
    assert "Dead" in sch.excluded and "Dead" not in sch.sensors


def test_describe_intensity_reports_every_class(op_pair):
    _, failures, _ = op_pair
    table = schema.describe_intensity(failures, "Intensity", "Failure")
    assert len(table) == failures["Failure"].nunique()
    assert (table["abs_max"] > 0).all()


def test_guess_intensity_mode():
    assert schema.guess_intensity_mode(np.linspace(0, 100, 50)) == "percent"
    assert schema.guess_intensity_mode(np.linspace(0, 5000, 50)) == "absolute"


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------

def test_correlation_matrix_is_bounded_and_unit_diagonal(op_pair):
    nominal, _, truth = op_pair
    R = quality.correlation_matrix(nominal[truth["sensors"]].to_numpy())
    assert np.allclose(np.diag(R), 1.0)
    assert np.abs(R).max() <= 1.0 + 1e-12


def test_check_finds_the_planted_duplicate(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    rep = quality.check(nominal, sensors)
    pairs = {frozenset((a, b)) for a, b, _ in rep.duplicate_pairs}
    assert frozenset((truth["sensors"][0], truth["sensors"][0] + "_bis")) in pairs
    assert rep.effective_rank < rep.n_sensors
    assert any("duplicated" in w for w in rep.warnings)


def test_check_finds_the_planted_collinear_pair(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    rep = quality.check(nominal, sensors)
    near = {frozenset((a, b)) for a, b, _ in rep.collinear_pairs}
    assert frozenset((truth["sensors"][1], truth["sensors"][1] + "_red")) in near


def test_check_flags_constant_sensor(op_pair):
    nominal, _, truth = op_pair
    df = nominal.copy()
    df["Dead"] = 3.0
    rep = quality.check(df, truth["sensors"] + ["Dead"])
    assert rep.constant == ["Dead"]
    assert any("constant" in w for w in rep.warnings)


def test_redundant_sensors_keeps_one_of_each_pair(op_pair):
    nominal, _, truth = op_pair
    sensors = truth["sensors"] + truth["extra_sensors"]
    rep = quality.check(nominal, sensors)
    drop = rep.redundant_sensors()
    assert set(drop) == set(truth["extra_sensors"])


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------

def test_standardizer_standard(op_pair):
    nominal, _, truth = op_pair
    Z = preprocess.Standardizer("standard").fit_transform(nominal[truth["sensors"]])
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_standardizer_robust_is_insensitive_to_outliers(op_pair):
    nominal, _, truth = op_pair
    clean = nominal[truth["sensors"]].copy()
    dirty = clean.copy()
    dirty.iloc[:20] = dirty.iloc[:20] * 50.0  # heavy contamination

    robust_clean = preprocess.Standardizer("robust").fit(clean)
    robust_dirty = preprocess.Standardizer("robust").fit(dirty)
    plain_dirty = preprocess.Standardizer("standard").fit(dirty)
    plain_clean = preprocess.Standardizer("standard").fit(clean)

    robust_shift = np.abs(robust_dirty.scale_ / robust_clean.scale_ - 1.0).max()
    plain_shift = np.abs(plain_dirty.scale_ / plain_clean.scale_ - 1.0).max()
    assert robust_shift < plain_shift


def test_standardizer_handles_zero_scale():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "dead": [5.0, 5.0, 5.0]})
    sc = preprocess.Standardizer("standard").fit(df)
    Z = sc.transform(df)
    assert np.all(np.isfinite(Z))
    assert sc.degenerate_ == ["dead"]


def test_standardizer_matches_columns_by_name(op_pair):
    nominal, _, truth = op_pair
    cols = truth["sensors"]
    sc = preprocess.Standardizer("standard").fit(nominal[cols])
    shuffled = nominal[list(reversed(cols))]
    assert np.allclose(sc.transform(shuffled), sc.transform(nominal[cols]))


def test_drop_redundant_removes_duplicates_and_constants(op_pair):
    nominal, _, truth = op_pair
    df = nominal.copy()
    df["Dead"] = 7.0
    sensors = truth["sensors"] + truth["extra_sensors"] + ["Dead"]
    kept, dropped = preprocess.drop_redundant(df[sensors], sensors)
    assert set(kept) == set(truth["sensors"])
    assert dropped["Dead"] == "(constant)"
    assert dropped[truth["sensors"][0] + "_bis"] == truth["sensors"][0]


def test_add_noise_is_reproducible_and_scaled(op_pair):
    nominal, _, truth = op_pair
    X = nominal[truth["sensors"]]
    a = preprocess.add_noise(X, 0.5, mode="sigma", random_state=1)
    b = preprocess.add_noise(X, 0.5, mode="sigma", random_state=1)
    c = preprocess.add_noise(X, 0.5, mode="sigma", random_state=2)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    grown = a.std(axis=0) / X.to_numpy().std(axis=0)
    assert np.all(grown > 1.0)  # variance strictly increased
    assert np.allclose(preprocess.add_noise(X, 0.0), X.to_numpy())
