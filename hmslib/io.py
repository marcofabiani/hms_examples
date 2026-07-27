"""Data discovery and loading.

Layout assumed by the project: for every operating point (OP) there is one CSV
with the nominal Monte Carlo cloud and one CSV with the failure runs.  File and
column names are not known in advance, so the flow is:

    hmslib.io.scan_folder("data/", write="manifest.json")   # once
    # open manifest.json, fix anything the scan got wrong
    ds = hmslib.Dataset.from_manifest("manifest.json")

:func:`scan_folder` only ever *proposes* a pairing; the manifest is the single
source of truth afterwards and is meant to be edited by hand.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import quality, schema as schema_mod
from .config import get_rng
from .schema import ColumnSchema

__all__ = [
    "scan_folder",
    "load_manifest",
    "save_manifest",
    "load_csv",
    "OperatingPointData",
    "Dataset",
    "split_frame",
    "stratified_split",
]


NOMINAL_TOKENS = (
    "nominal", "nominale", "nom", "base", "baseline", "ref", "reference",
    "healthy", "sano", "ok",
)
FAILURE_TOKENS = (
    "failures", "failure", "fail", "faults", "fault", "guasti", "guasto",
    "anomaly", "anomalies", "anomal", "degraded", "ko",
)


def _tokens(stem: str) -> List[str]:
    """Split a file stem into tokens, keeping the original case.

    Case is preserved so that ``OP_100_nominal.csv`` yields the operating point
    ``OP_100`` rather than ``op_100``; comparisons against the token
    vocabularies below are done case-insensitively.
    """
    return [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t]


_IGNORED_TOKENS = ("table", "data", "csv")


def _classify_file(path: str) -> Optional[str]:
    """Return ``'nominal'``, ``'failures'`` or ``None`` from the file name."""
    toks = [t.lower() for t in _tokens(os.path.splitext(os.path.basename(path))[0])]
    # failure first: 'nominal_vs_failures' is a failure table
    for t in toks:
        if t in FAILURE_TOKENS:
            return "failures"
    for t in toks:
        if t in NOMINAL_TOKENS:
            return "nominal"
    for t in toks:  # substring fallback, e.g. 'tablefailures'
        if any(f in t for f in ("failure", "fault", "guast")):
            return "failures"
        if any(n in t for n in ("nominal", "nominale")):
            return "nominal"
    return None


def _op_key(path: str, kind: str) -> str:
    """Best effort operating point key from a file name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    role = NOMINAL_TOKENS if kind == "nominal" else FAILURE_TOKENS
    rest = [t for t in _tokens(stem)
            if t.lower() not in role and t.lower() not in _IGNORED_TOKENS]
    return "_".join(rest)


def scan_folder(
    root: str,
    recursive: bool = True,
    write: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Propose a manifest for a folder of CSVs.

    Pairing strategy, in order:

    1. group the CSVs by the directory that contains them; if a directory holds
       exactly one nominal and one failure file, they are paired and the OP is
       named after the directory (or after the common part of the file names
       when the directory is the scan root);
    2. otherwise, pair files whose name reduces to the same key once the
       nominal/failure token is removed.

    Anything left unpaired is listed under ``unmatched`` in the manifest, so
    nothing is silently dropped.
    """
    root = os.path.abspath(root)
    files: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".csv"):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in sorted(os.listdir(root)):
            if fn.lower().endswith(".csv"):
                files.append(os.path.join(root, fn))
    files.sort()

    by_dir: Dict[str, List[str]] = {}
    for f in files:
        by_dir.setdefault(os.path.dirname(f), []).append(f)

    ops: Dict[str, Dict[str, str]] = {}
    unmatched: List[str] = []
    notes: List[str] = []

    for dirpath, group in sorted(by_dir.items()):
        kinds = {f: _classify_file(f) for f in group}
        noms = [f for f, k in kinds.items() if k == "nominal"]
        fails = [f for f, k in kinds.items() if k == "failures"]
        unknown = [f for f, k in kinds.items() if k is None]
        unmatched.extend(unknown)

        if len(noms) == 1 and len(fails) == 1:
            name = _dir_op_name(dirpath, root, noms[0], fails[0])
            ops[_unique_key(ops, name)] = {
                "nominal": os.path.relpath(noms[0], root).replace("\\", "/"),
                "failures": os.path.relpath(fails[0], root).replace("\\", "/"),
            }
            continue

        keyed: Dict[str, Dict[str, str]] = {}
        for f in noms:
            keyed.setdefault(_op_key(f, "nominal"), {})["nominal"] = f
        for f in fails:
            keyed.setdefault(_op_key(f, "failures"), {})["failures"] = f
        for key, pair in sorted(keyed.items()):
            if "nominal" in pair and "failures" in pair:
                name = _unique_key(ops, key or os.path.basename(dirpath) or "OP")
                ops[name] = {
                    "nominal": os.path.relpath(pair["nominal"], root).replace("\\", "/"),
                    "failures": os.path.relpath(pair["failures"], root).replace("\\", "/"),
                }
            else:
                unmatched.extend(pair.values())
                notes.append(
                    "in %s the key %r has only the %s file"
                    % (os.path.relpath(dirpath, root) or ".", key, list(pair)[0])
                )

    manifest: Dict[str, Any] = {
        "root": root.replace("\\", "/"),
        "operating_points": ops,
        "columns": {"label": None, "intensity": None, "exclude": []},
        "intensity_mode": "auto",
        "unmatched": [os.path.relpath(f, root).replace("\\", "/") for f in sorted(set(unmatched))],
        "notes": notes,
    }

    if verbose:
        print("scan_folder: %s" % root)
        print("  %d CSV file(s), %d operating point(s) proposed" % (len(files), len(ops)))
        for name, pair in ops.items():
            print("    %-20s nominal=%s  failures=%s"
                  % (name, pair["nominal"], pair["failures"]))
        for f in manifest["unmatched"]:
            print("    ! unmatched: %s" % f)
        for n in notes:
            print("    ! %s" % n)
        if not ops:
            print("    no pair found: write the manifest by hand "
                  "(see hmslib.io.save_manifest)")

    if write:
        save_manifest(manifest, write)
        if verbose:
            print("  manifest written to %s -- review it before loading" % write)
    return manifest


_GENERIC_FOLDER_NAMES = ("data", "dataset", "csv", "input", "inputs", "raw")


def _dir_op_name(dirpath: str, root: str, nominal: str, failures: str) -> str:
    rel = os.path.relpath(dirpath, root)
    if rel not in (".", ""):
        return rel.replace("\\", "_").replace("/", "_")
    common = _common_key(nominal, failures)
    if common:
        return common
    base = os.path.basename(root)
    if base and base.lower() not in _GENERIC_FOLDER_NAMES:
        return base
    return "OP"


def _common_key(nominal: str, failures: str) -> str:
    """Tokens shared by the two file names, i.e. what identifies the OP."""
    nom_tokens = _tokens(os.path.splitext(os.path.basename(nominal))[0])
    fail_lower = {t.lower() for t in _tokens(os.path.splitext(os.path.basename(failures))[0])}
    shared = [t for t in nom_tokens
              if t.lower() in fail_lower
              and t.lower() not in NOMINAL_TOKENS
              and t.lower() not in FAILURE_TOKENS
              and t.lower() not in _IGNORED_TOKENS]
    return "_".join(shared)


def _unique_key(existing: Dict[str, Any], name: str) -> str:
    name = name or "OP"
    if name not in existing:
        return name
    i = 2
    while "%s_%d" % (name, i) in existing:
        i += 1
    return "%s_%d" % (name, i)


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Write a manifest as indented JSON."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if "root" not in manifest or not manifest["root"]:
        manifest["root"] = os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(manifest["root"]):
        manifest["root"] = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(path)), manifest["root"])
        )
    return manifest


def load_csv(
    path: str, drop_non_finite: bool = True, verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a CSV, turn infinities into NaN and optionally drop bad rows.

    Returns ``(frame, info)``; ``info`` records how many rows were removed, so
    that silent data loss is impossible to miss.
    """
    df = pd.read_csv(path)
    n0 = len(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    n_bad = int(df[num_cols].isna().any(axis=1).sum()) if num_cols else 0
    if drop_non_finite and n_bad:
        df = df.dropna(subset=num_cols).reset_index(drop=True)
    info = {"path": path, "rows_read": n0, "rows_dropped": n0 - len(df),
            "rows_with_non_finite": n_bad}
    if verbose and info["rows_dropped"]:
        print("  %s: dropped %d/%d rows with NaN or inf"
              % (os.path.basename(path), info["rows_dropped"], n0))
    return df, info


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

@dataclass
class OperatingPointData:
    """Nominal cloud plus failure runs for a single operating point."""

    name: str
    nominal: pd.DataFrame
    failures: pd.DataFrame
    schema: ColumnSchema
    load_info: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def sensors(self) -> List[str]:
        return list(self.schema.sensors)

    @property
    def label(self) -> Optional[str]:
        return self.schema.label

    @property
    def intensity(self) -> Optional[str]:
        return self.schema.intensity

    @property
    def classes(self) -> List[str]:
        if self.label is None or self.label not in self.failures.columns:
            return []
        return sorted(str(v) for v in self.failures[self.label].dropna().unique())

    def X_nominal(self) -> pd.DataFrame:
        return self.nominal[self.sensors]

    def X_failures(self, classes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return self.failures_subset(classes)[self.sensors]

    def failures_subset(self, classes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if classes is None or self.label is None:
            return self.failures
        wanted = set(map(str, classes))
        mask = self.failures[self.label].astype(str).isin(wanted)
        return self.failures.loc[mask]

    def intensity_values(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Signed intensity of each failure row (NaN if no intensity column)."""
        df = self.failures if df is None else df
        if self.intensity is None or self.intensity not in df.columns:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[self.intensity], errors="coerce").to_numpy(dtype=float)

    def class_counts(self) -> pd.Series:
        if self.label is None:
            return pd.Series(dtype=int)
        return self.failures[self.label].astype(str).value_counts().sort_index()

    def quality(self, which: str = "nominal") -> quality.QualityReport:
        df = self.nominal if which == "nominal" else self.failures
        return quality.check(df, self.sensors)

    def describe_intensity(self) -> pd.DataFrame:
        if self.intensity is None:
            return pd.DataFrame()
        return schema_mod.describe_intensity(self.failures, self.intensity, self.label)

    def summary(self) -> str:
        lines = [
            "OperatingPoint %r" % self.name,
            "  nominal  : %d rows" % len(self.nominal),
            "  failures : %d rows, %d class(es)" % (len(self.failures), len(self.classes)),
            "  sensors  : %d" % len(self.sensors),
            "  intensity: %s" % (self.intensity or "-- none --"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "OperatingPointData(%r, nominal=%d, failures=%d, sensors=%d)" % (
            self.name, len(self.nominal), len(self.failures), len(self.sensors),
        )


class Dataset:
    """All operating points, loaded and schema-resolved."""

    def __init__(
        self,
        ops: Dict[str, OperatingPointData],
        manifest: Optional[Dict[str, Any]] = None,
        intensity_mode: str = "auto",
    ):
        self._ops = dict(ops)
        self.manifest = manifest or {}
        self.intensity_mode = intensity_mode

    # -- construction -----------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        path_or_manifest: Any,
        verbose: bool = True,
        drop_non_finite: bool = True,
    ) -> "Dataset":
        manifest = (
            load_manifest(path_or_manifest)
            if isinstance(path_or_manifest, str)
            else dict(path_or_manifest)
        )
        root = manifest.get("root") or "."
        columns = manifest.get("columns") or {}
        label = columns.get("label")
        intensity = columns.get("intensity")
        exclude = columns.get("exclude") or []
        sensors_override = columns.get("sensors")

        ops: Dict[str, OperatingPointData] = {}
        for name, pair in (manifest.get("operating_points") or {}).items():
            nom_path = _resolve(root, pair["nominal"])
            fail_path = _resolve(root, pair["failures"])
            nominal, info_n = load_csv(nom_path, drop_non_finite, verbose)
            failures, info_f = load_csv(fail_path, drop_non_finite, verbose)
            sch = schema_mod.infer_schema(
                failures, nominal, label=label, intensity=intensity, exclude=exclude,
            )
            if sensors_override:
                keep = [c for c in sensors_override if c in sch.sensors]
                dropped = [c for c in sch.sensors if c not in keep]
                sch.sensors = keep
                sch.excluded.extend(dropped)
                sch.notes.append("sensor list overridden by the manifest")
            ops[name] = OperatingPointData(
                name=name, nominal=nominal, failures=failures, schema=sch,
                load_info={"nominal": info_n, "failures": info_f},
            )

        ds = cls(ops, manifest, manifest.get("intensity_mode", "auto"))
        if verbose:
            ds.describe()
        return ds

    @classmethod
    def from_folder(cls, root: str, verbose: bool = True, **kwargs: Any) -> "Dataset":
        """Scan and load in one step, without writing a manifest to disk."""
        manifest = scan_folder(root, write=None, verbose=verbose)
        for key in ("label", "intensity", "exclude"):
            if key in kwargs:
                manifest["columns"][key] = kwargs.pop(key)
        return cls.from_manifest(manifest, verbose=verbose, **kwargs)

    @classmethod
    def from_frames(
        cls,
        nominal: pd.DataFrame,
        failures: pd.DataFrame,
        name: str = "OP",
        **schema_kwargs: Any,
    ) -> "Dataset":
        """Build a single-OP dataset from in-memory frames (tests, quick tries)."""
        sch = schema_mod.infer_schema(failures, nominal, **schema_kwargs)
        op = OperatingPointData(name, nominal, failures, sch)
        return cls({name: op})

    # -- access -----------------------------------------------------------
    @property
    def operating_points(self) -> List[str]:
        return list(self._ops)

    def __getitem__(self, name: str) -> OperatingPointData:
        if name not in self._ops:
            raise KeyError(
                "unknown operating point %r; available: %s" % (name, self.operating_points)
            )
        return self._ops[name]

    def __iter__(self) -> Iterator[OperatingPointData]:
        return iter(self._ops.values())

    def __len__(self) -> int:
        return len(self._ops)

    def items(self) -> Iterator[Tuple[str, OperatingPointData]]:
        return iter(self._ops.items())

    def common_sensors(self) -> List[str]:
        """Sensors available in *every* operating point.

        Some sensors may be disabled at some operating points; per-OP models do
        not care, but any cross-OP comparison must use this list.
        """
        if not self._ops:
            return []
        sets = [set(op.sensors) for op in self._ops.values()]
        common = set.intersection(*sets)
        first = next(iter(self._ops.values())).sensors
        return [s for s in first if s in common]

    def sensor_availability(self) -> pd.DataFrame:
        """Boolean table sensor x operating point."""
        all_sensors: List[str] = []
        for op in self._ops.values():
            for s in op.sensors:
                if s not in all_sensors:
                    all_sensors.append(s)
        data = {name: [s in set(op.sensors) for s in all_sensors]
                for name, op in self._ops.items()}
        return pd.DataFrame(data, index=all_sensors)

    # -- reporting --------------------------------------------------------
    def summary(self) -> str:
        lines = ["Dataset: %d operating point(s)" % len(self._ops)]
        for name, op in self._ops.items():
            lines.append(
                "  %-18s nominal=%6d  failures=%6d  sensors=%3d  classes=%3d"
                % (name, len(op.nominal), len(op.failures), len(op.sensors),
                   len(op.classes))
            )
        common = self.common_sensors()
        if len(self._ops) > 1:
            lines.append("  sensors common to all OPs: %d" % len(common))
            avail = self.sensor_availability()
            missing = avail.index[~avail.all(axis=1)].tolist()
            if missing:
                lines.append("  ! not available everywhere: %s" % missing)
        first = next(iter(self._ops.values()), None)
        if first is not None:
            lines.append("")
            lines.append(first.schema.summary())
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Dataset(%s)" % ", ".join(self.operating_points)


def _resolve(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def split_frame(
    df: pd.DataFrame, test_size: float = 0.3, random_state: Any = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plain random row split (used for the nominal cloud)."""
    rng = get_rng(random_state)
    idx = rng.permutation(len(df))
    n_test = int(round(test_size * len(df)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def stratified_split(
    df: pd.DataFrame,
    label: Optional[str] = None,
    intensity: Optional[str] = None,
    test_size: float = 0.3,
    n_bins: int = 5,
    random_state: Any = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split failure runs stratifying by class *and* intensity bin.

    With a continuous intensity sweep, a purely random split can put most of the
    high intensity runs on one side and inflate the reported performance.
    Binning the intensity by quantiles within each class and splitting each bin
    keeps the two sides comparable.

    Strata with a single row go entirely to the training side.
    """
    rng = get_rng(random_state)
    n = len(df)
    if n == 0:
        return df.copy(), df.copy()

    strata = pd.Series(["all"] * n, index=df.index)
    if label is not None and label in df.columns:
        strata = df[label].astype(str)
    if intensity is not None and intensity in df.columns and n_bins > 1:
        values = pd.to_numeric(df[intensity], errors="coerce").abs()
        bins = pd.Series(np.zeros(n, dtype=int), index=df.index)
        for key in strata.unique():
            mask = (strata == key).to_numpy()
            v = values[mask]
            try:
                b = pd.qcut(v.rank(method="first"), q=min(n_bins, max(int(mask.sum()), 1)),
                            labels=False, duplicates="drop")
            except (ValueError, IndexError):
                b = pd.Series(np.zeros(int(mask.sum()), dtype=int), index=v.index)
            bins.loc[v.index] = np.asarray(b, dtype=int)
        strata = strata.astype(str) + "|" + bins.astype(str)

    train_pos: List[int] = []
    test_pos: List[int] = []
    positions = np.arange(n)
    for key in pd.unique(strata):
        pos = positions[(strata == key).to_numpy()]
        if pos.size < 2:
            train_pos.extend(pos.tolist())
            continue
        perm = rng.permutation(pos.size)
        n_test = int(round(test_size * pos.size))
        n_test = min(max(n_test, 0), pos.size - 1)
        test_pos.extend(pos[perm[:n_test]].tolist())
        train_pos.extend(pos[perm[n_test:]].tolist())

    train_pos.sort()
    test_pos.sort()
    return df.iloc[train_pos].copy(), df.iloc[test_pos].copy()
