"""Shared test fixtures.

The tests never touch the real data: everything runs on
:mod:`hmslib.synth`, whose ground truth (covariance, fault directions,
deliberate rank deficiency) is known by construction.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display on the target machine

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hmslib as hm  # noqa: E402


@pytest.fixture(scope="session")
def op_pair():
    """A single operating point: (nominal, failures, truth)."""
    return hm.synth.make_operating_point(
        n_sensors=10, n_nominal=1500, n_classes=4, n_per_class=250,
        random_state=7,
    )


@pytest.fixture(scope="session")
def dataset():
    """Two operating points, in memory."""
    ds, truths = hm.synth.make_dataset(
        n_ops=2, n_sensors=8, n_nominal=1200, n_classes=3, n_per_class=200,
        random_state=11,
    )
    return ds, truths


@pytest.fixture(scope="session")
def gaussian_clean():
    """Well conditioned Gaussian data with a known covariance.

    Returns ``(train, test, cov)`` with 8 features, enough samples to make the
    asymptotic chi-square results testable.
    """
    rng = np.random.RandomState(3)
    p = 8
    A = rng.normal(size=(p, p))
    cov = A @ A.T + np.eye(p) * 0.5
    mean = rng.uniform(10.0, 20.0, size=p)
    train = rng.multivariate_normal(mean, cov, size=20000)
    test = rng.multivariate_normal(mean, cov, size=20000)
    return train, test, cov


@pytest.fixture
def tmp_csv_folder(tmp_path):
    """A folder on disk holding two operating points as CSVs."""
    root = str(tmp_path / "data")
    hm.synth.write_dataset(
        root, n_ops=2, n_sensors=6, n_nominal=400, n_classes=2, n_per_class=120,
        random_state=5,
    )
    return root
