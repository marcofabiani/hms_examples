"""hmslib - statistics and fault detection & diagnosis for liquid rocket engines.

Designed to run on an offline machine: no dependency beyond numpy, scipy,
pandas, scikit-learn and matplotlib (torch only for :mod:`hmslib.nn`), and no
installation step - dropping the package folder next to your scripts is enough.

Typical session
---------------
>>> import hmslib as hm
>>> hm.check_env()
>>> hm.io.scan_folder("data/", write="manifest.json")   # once, then edit by hand
>>> ds = hm.Dataset.from_manifest("manifest.json")
>>> hm.quicklook(ds, out="reports/")
>>> det = hm.Mahalanobis(cov="ledoit_wolf", threshold="empirical", alpha=1e-3)
>>> bank = hm.ModelBank.fit(ds, det)
>>> bank["OP_100"].diagnostics_
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import analysis, compat, config, io, preprocess, quality, schema, synth, viz
from .analysis import PODResult, fit_pod, pod_analysis, pod_table, sensor_sensitivity
from .compat import check_env
from .config import PLOT, apply_style, set_seed
from .detect import Mahalanobis
from .detect.base import BaseDetector
from .bank import ModelBank
from .io import Dataset, OperatingPointData, load_csv, scan_folder
from .preprocess import Standardizer, add_noise, drop_redundant
from .quality import QualityReport
from .report import detection_report, quicklook
from .schema import ColumnSchema, infer_schema

__all__ = [
    "__version__",
    # modules
    "analysis", "compat", "config", "io", "preprocess", "quality", "schema",
    "synth", "viz",
    # environment / style
    "check_env", "apply_style", "set_seed", "PLOT",
    # data
    "Dataset", "OperatingPointData", "scan_folder", "load_csv",
    "ColumnSchema", "infer_schema", "QualityReport",
    "Standardizer", "drop_redundant", "add_noise",
    # models
    "BaseDetector", "Mahalanobis", "ModelBank",
    # detectability
    "PODResult", "fit_pod", "pod_analysis", "pod_table", "sensor_sensitivity",
    # reporting
    "quicklook", "detection_report",
]
