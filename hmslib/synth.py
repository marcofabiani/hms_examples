"""Synthetic data generator.

Two uses: exercising the library before the real data arrive, and giving the
test suite data whose properties are known by construction (true covariance,
true fault directions, deliberate rank deficiency).

The generator mimics the structure described for the project: a Monte Carlo
cloud around a nominal operating point, and failure runs where the fault
intensity is swept *inside* the Monte Carlo loop, so each run has both its own
intensity and its own realisation of the nominal scatter.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import get_rng

__all__ = ["make_operating_point", "make_dataset", "write_dataset"]


def _sensor_names(n: int) -> List[str]:
    families = ["Pc", "Pout", "Tin", "Tout", "w", "m", "eta", "N"]
    names: List[str] = []
    i = 0
    while len(names) < n:
        fam = families[i % len(families)]
        idx = i // len(families) + 1
        names.append("%s_%d" % (fam, idx))
        i += 1
    return names[:n]


def _covariance(
    n_sensors: int, n_factors: int, rng: np.random.RandomState,
) -> np.ndarray:
    """Low rank factor model plus diagonal noise: correlated but full rank."""
    loadings = rng.normal(0.0, 1.0, size=(n_sensors, n_factors))
    cov = loadings @ loadings.T
    cov += np.diag(rng.uniform(0.2, 0.6, size=n_sensors))
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)  # unit variances, easier to reason about


def make_operating_point(
    n_sensors: int = 12,
    n_nominal: int = 1500,
    n_classes: int = 6,
    n_per_class: int = 400,
    n_factors: int = 4,
    duplicate_sensor: bool = True,
    collinear_sensor: bool = True,
    constant_sensor: bool = False,
    intensity_max: float = 40.0,
    effect_scale: float = 0.12,
    n_affected: int = 3,
    scale_shift: float = 0.0,
    random_state: Any = 0,
    class_prefix: str = "FAULT",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate one operating point.

    Returns ``(nominal, failures, truth)``.  ``truth`` records the quantities a
    test can check against: the covariance actually used, the fault direction of
    every class and the sensors it touches.

    Parameters worth knowing
    ------------------------
    duplicate_sensor : append an exact copy of one sensor, making the covariance
        singular - the pathology observed on the real SSME tables.
    collinear_sensor : append a sensor equal to another one plus a tiny noise,
        producing a near singular direction.
    effect_scale : deviation in nominal sigmas caused by one unit of intensity,
        so a class with intensity ``i`` sits about ``effect_scale * i`` sigmas
        away along its own direction.
    scale_shift : shifts the operating point (added to every mean), used to make
        several distinguishable operating points.
    """
    rng = get_rng(random_state)
    names = _sensor_names(n_sensors)

    mean = 100.0 * rng.uniform(0.5, 2.0, size=n_sensors) + scale_shift
    sigma = np.abs(mean) * rng.uniform(0.005, 0.02, size=n_sensors)
    corr = _covariance(n_sensors, n_factors, rng)
    cov = corr * np.outer(sigma, sigma)

    nominal = rng.multivariate_normal(mean, cov, size=n_nominal)

    directions: Dict[str, np.ndarray] = {}
    affected: Dict[str, List[str]] = {}
    rows: List[np.ndarray] = []
    labels: List[str] = []
    intensities: List[float] = []

    for c in range(n_classes):
        cls = "%s_%02d" % (class_prefix, c + 1)
        idx = rng.choice(n_sensors, size=min(n_affected, n_sensors), replace=False)
        d = np.zeros(n_sensors)
        d[idx] = rng.choice([-1.0, 1.0], size=idx.size) * rng.uniform(0.5, 1.5, size=idx.size)
        d = d / np.linalg.norm(d)
        directions[cls] = d
        affected[cls] = [names[i] for i in np.sort(idx)]

        base = rng.multivariate_normal(mean, cov, size=n_per_class)
        # intensity swept inside the Monte Carlo loop: one value per run
        inten = rng.uniform(0.0, intensity_max, size=n_per_class)
        shift = (inten[:, None] * effect_scale) * (d * sigma)[None, :]
        rows.append(base + shift)
        labels.extend([cls] * n_per_class)
        intensities.extend(inten.tolist())

    failures = np.vstack(rows) if rows else np.empty((0, n_sensors))

    df_nom = pd.DataFrame(nominal, columns=names)
    df_fail = pd.DataFrame(failures, columns=names)

    extra_names: List[str] = []
    if duplicate_sensor and n_sensors >= 1:
        src = names[0]
        dup = src + "_bis"
        df_nom[dup] = df_nom[src]
        df_fail[dup] = df_fail[src]
        extra_names.append(dup)
    if collinear_sensor and n_sensors >= 2:
        src = names[1]
        col = src + "_red"
        # 2% of the sensor's own scatter, which puts the correlation at about
        # 0.9998: near singular, but not an exact duplicate
        jitter = float(df_nom[src].to_numpy().std()) * 0.02
        df_nom[col] = df_nom[src] + rng.normal(0.0, jitter, size=len(df_nom))
        df_fail[col] = df_fail[src] + rng.normal(0.0, jitter, size=len(df_fail))
        extra_names.append(col)
    if constant_sensor:
        df_nom["Const_1"] = 42.0
        df_fail["Const_1"] = 42.0
        extra_names.append("Const_1")

    df_fail["Failure"] = labels
    df_fail["Intensity"] = intensities

    truth = {
        "sensors": names,
        "extra_sensors": extra_names,
        "mean": mean,
        "sigma": sigma,
        "cov": cov,
        "directions": directions,
        "affected": affected,
        "effect_scale": effect_scale,
        "intensity_max": intensity_max,
    }
    return df_nom, df_fail, truth


def make_dataset(
    n_ops: int = 2,
    random_state: Any = 0,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    """Build an in-memory :class:`hmslib.io.Dataset` with ``n_ops`` operating points.

    Returns ``(dataset, truth_by_op)``.
    """
    from .io import Dataset, OperatingPointData
    from .schema import infer_schema

    rng = get_rng(random_state)
    ops = {}
    truths: Dict[str, Dict[str, Any]] = {}
    for k in range(n_ops):
        name = "OP_%d" % (60 + 20 * k)
        seed = int(rng.randint(0, 10 ** 6))
        nom, fail, truth = make_operating_point(
            random_state=seed, scale_shift=10.0 * k, **kwargs
        )
        sch = infer_schema(fail, nom)
        ops[name] = OperatingPointData(name, nom, fail, sch)
        truths[name] = truth
    return Dataset(ops), truths


def write_dataset(
    root: str,
    n_ops: int = 2,
    random_state: Any = 0,
    per_op_folder: bool = False,
    **kwargs: Any,
) -> str:
    """Write a synthetic dataset to disk as CSVs and return the folder path.

    Useful to rehearse the real workflow, ``scan_folder`` included.
    """
    rng = get_rng(random_state)
    os.makedirs(root, exist_ok=True)
    for k in range(n_ops):
        name = "OP_%d" % (60 + 20 * k)
        seed = int(rng.randint(0, 10 ** 6))
        nom, fail, _ = make_operating_point(
            random_state=seed, scale_shift=10.0 * k, **kwargs
        )
        folder = os.path.join(root, name) if per_op_folder else root
        os.makedirs(folder, exist_ok=True)
        prefix = "" if per_op_folder else name + "_"
        nom.to_csv(os.path.join(folder, "%snominal.csv" % prefix), index=False)
        fail.to_csv(os.path.join(folder, "%sfailures.csv" % prefix), index=False)
    return root
