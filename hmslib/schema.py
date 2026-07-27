"""Column role inference.

The incoming CSVs have unknown column names, so the library has to work out
which column is the failure label, which one carries the fault intensity and
which ones are sensors.  Every decision taken here is *proposed and printed*,
never applied silently: :meth:`ColumnSchema.describe` shows what was inferred
and why, and any field can be overridden explicitly.

The strongest signal is comparing the failure table against the nominal table
of the same operating point: label and intensity columns exist only in the
former.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["ColumnSchema", "infer_schema", "describe_intensity", "guess_intensity_mode"]


_INTENSITY_PATTERNS = (
    r"intens", r"sever", r"magnitud", r"amplitud", r"fault[_ ]?size",
    r"^level$", r"_level$", r"degrad", r"perc", r"pct", r"^size$",
)
_LABEL_PATTERNS = (
    r"fail", r"fault", r"class", r"label", r"mode", r"guast", r"anomal", r"type",
)
_META_PATTERNS = (
    r"^id$", r"_id$", r"^run$", r"^case$", r"^seed$", r"^index$", r"^unnamed",
    r"^sample$", r"^mc$", r"^iter",
)

# A column with at most this many distinct values is a candidate label even if
# it is numeric (failure modes are sometimes coded as integers).
_MAX_LABEL_CARDINALITY = 60


def _matches(name: str, patterns: Sequence[str]) -> bool:
    low = str(name).strip().lower()
    return any(re.search(p, low) for p in patterns)


@dataclass
class ColumnSchema:
    """Roles assigned to the columns of a failure/nominal table pair."""

    sensors: List[str] = field(default_factory=list)
    label: Optional[str] = None
    intensity: Optional[str] = None
    meta: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensors": list(self.sensors),
            "label": self.label,
            "intensity": self.intensity,
            "meta": list(self.meta),
            "excluded": list(self.excluded),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSchema":
        return cls(
            sensors=list(data.get("sensors") or []),
            label=data.get("label"),
            intensity=data.get("intensity"),
            meta=list(data.get("meta") or []),
            excluded=list(data.get("excluded") or []),
        )

    # -- reporting --------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for name in self.sensors:
            rows.append({"column": name, "role": "sensor"})
        if self.label:
            rows.append({"column": self.label, "role": "label"})
        if self.intensity:
            rows.append({"column": self.intensity, "role": "intensity"})
        for name in self.meta:
            rows.append({"column": name, "role": "meta"})
        for name in self.excluded:
            rows.append({"column": name, "role": "excluded"})
        return pd.DataFrame(rows, columns=["column", "role"])

    def summary(self) -> str:
        lines = [
            "ColumnSchema",
            "  sensors   : %d  %s" % (len(self.sensors), _preview(self.sensors)),
            "  label     : %s" % (self.label or "-- none --"),
            "  intensity : %s" % (self.intensity or "-- none --"),
        ]
        if self.meta:
            lines.append("  meta      : %s" % _preview(self.meta))
        if self.excluded:
            lines.append("  excluded  : %s" % _preview(self.excluded))
        if self.notes:
            lines.append("  notes:")
            lines.extend("    - " + n for n in self.notes)
        if self.ambiguities:
            lines.append("  AMBIGUOUS (override explicitly if wrong):")
            lines.extend("    ! " + a for a in self.ambiguities)
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ColumnSchema(sensors=%d, label=%r, intensity=%r)" % (
            len(self.sensors), self.label, self.intensity,
        )


def _preview(names: Sequence[str], k: int = 6) -> str:
    names = list(names)
    if len(names) <= k:
        return ", ".join(map(str, names))
    return ", ".join(map(str, names[:k])) + ", ... (+%d)" % (len(names) - k)


def infer_schema(
    failures: pd.DataFrame,
    nominal: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    intensity: Optional[str] = None,
    exclude: Optional[Sequence[str]] = None,
    drop_constant: bool = True,
) -> ColumnSchema:
    """Infer column roles for a (failure, nominal) table pair.

    Parameters
    ----------
    failures : table containing the failure runs (label and intensity live here).
    nominal  : matching nominal table; when given, columns present only in
               ``failures`` become the prime candidates for label/intensity.
    label, intensity : explicit overrides.  Pass a name to force it, or the
               string ``"none"`` to state that the column does not exist.
    exclude  : columns to remove from consideration entirely.
    drop_constant : move zero-variance numeric columns to ``excluded`` instead
               of treating them as sensors (a constant sensor breaks any
               covariance based method and is never informative).
    """
    schema = ColumnSchema()
    exclude = list(exclude or [])
    cols = [c for c in failures.columns if c not in exclude]
    schema.excluded.extend([c for c in failures.columns if c in exclude])

    extra: List[str] = []
    if nominal is not None:
        nom_cols = set(nominal.columns)
        extra = [c for c in cols if c not in nom_cols]
        if extra:
            schema.notes.append(
                "columns present in the failure table only: %s" % _preview(extra)
            )
        else:
            schema.notes.append(
                "failure and nominal tables share all columns; label/intensity "
                "inferred from names alone"
            )

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(failures[c])]
    non_numeric = [c for c in cols if c not in numeric]

    # -- label ------------------------------------------------------------
    if label is not None:
        schema.label = None if str(label).lower() == "none" else label
        if schema.label is not None:
            schema.notes.append("label set explicitly to %r" % schema.label)
    else:
        cands = _label_candidates(failures, cols, non_numeric, extra)
        schema.label = cands[0] if cands else None
        if schema.label is None:
            schema.notes.append("no label column found")
        else:
            schema.notes.append("label inferred: %r" % schema.label)
        if len(cands) > 1:
            schema.ambiguities.append(
                "several label candidates %s -> chose %r" % (cands, schema.label)
            )

    # -- intensity --------------------------------------------------------
    if intensity is not None:
        schema.intensity = None if str(intensity).lower() == "none" else intensity
        if schema.intensity is not None:
            schema.notes.append("intensity set explicitly to %r" % schema.intensity)
    else:
        cands = _intensity_candidates(failures, numeric, extra, schema.label)
        schema.intensity = cands[0] if cands else None
        if schema.intensity is None:
            schema.notes.append(
                "no intensity column found; detectability/POD analyses will be "
                "unavailable until one is set"
            )
        else:
            schema.notes.append("intensity inferred: %r" % schema.intensity)
        if len(cands) > 1:
            schema.ambiguities.append(
                "several intensity candidates %s -> chose %r" % (cands, schema.intensity)
            )

    # -- meta / sensors ---------------------------------------------------
    taken = {schema.label, schema.intensity}
    for c in cols:
        if c in taken:
            continue
        if c in non_numeric or _matches(c, _META_PATTERNS):
            schema.meta.append(c)
            continue
        if drop_constant:
            values = pd.to_numeric(failures[c], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) == 0 or float(np.nanstd(finite.to_numpy(dtype=float))) == 0.0:
                schema.excluded.append(c)
                schema.notes.append("%r excluded: constant or all-NaN" % c)
                continue
        schema.sensors.append(c)

    if nominal is not None:
        missing = [c for c in schema.sensors if c not in nominal.columns]
        if missing:
            schema.sensors = [c for c in schema.sensors if c not in missing]
            schema.excluded.extend(missing)
            schema.notes.append(
                "%d sensor(s) absent from the nominal table, excluded: %s"
                % (len(missing), _preview(missing))
            )
        only_nominal = [
            c for c in nominal.columns
            if c not in failures.columns and pd.api.types.is_numeric_dtype(nominal[c])
        ]
        if only_nominal:
            schema.notes.append(
                "%d column(s) present in the nominal table only, ignored: %s"
                % (len(only_nominal), _preview(only_nominal))
            )

    return schema


def _label_candidates(
    df: pd.DataFrame, cols: Sequence[str], non_numeric: Sequence[str],
    extra: Sequence[str],
) -> List[str]:
    """Rank plausible label columns, best first."""
    scored = []
    for c in cols:
        n_unique = int(df[c].nunique(dropna=True))
        if n_unique < 2 or n_unique > _MAX_LABEL_CARDINALITY:
            continue
        score = 0.0
        if c in non_numeric:
            score += 3.0          # a string column with few values is the usual case
        if _matches(c, _LABEL_PATTERNS):
            score += 3.0
        if c in extra:
            score += 2.0          # absent from the nominal table
        if n_unique <= 40:
            score += 1.0
        if score < 3.0:
            continue
        scored.append((score, -n_unique, c))
    scored.sort(reverse=True)
    return [c for _, _, c in scored]


def _intensity_candidates(
    df: pd.DataFrame, numeric: Sequence[str], extra: Sequence[str],
    label: Optional[str],
) -> List[str]:
    """Rank plausible intensity columns, best first."""
    scored = []
    n_rows = max(len(df), 1)
    for c in numeric:
        if c == label:
            continue
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 2 or float(np.nanstd(finite)) == 0.0:
            continue
        score = 0.0
        if _matches(c, _INTENSITY_PATTERNS):
            score += 4.0
        if c in extra:
            score += 3.0          # only failure runs have an intensity
        # a swept parameter is continuous: many distinct values
        if finite.size and (len(np.unique(finite)) / n_rows) > 0.5:
            score += 1.0
        if score < 3.0:
            continue
        scored.append((score, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored]


# --------------------------------------------------------------------------
# intensity units
# --------------------------------------------------------------------------

def guess_intensity_mode(values: np.ndarray) -> str:
    """Guess whether an intensity column is a percentage or an absolute value.

    Returns ``'percent'``, ``'absolute'`` or ``'unknown'``.  This is a *hint*
    printed to the user, never applied automatically: the caller decides via
    ``intensity_mode``.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "unknown"
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo >= 0.0 and hi <= 100.0 and hi >= 20.0:
        return "percent"
    if hi > 100.0 or lo < -100.0:
        return "absolute"
    if lo >= 0.0 and hi <= 1.0:
        return "percent"  # fraction rather than percentage; still relative
    return "unknown"


def describe_intensity(
    df: pd.DataFrame, intensity: str, label: Optional[str] = None,
) -> pd.DataFrame:
    """Per class summary of the intensity column, plus a units guess.

    The sign of the intensity is only meaningful for physical interpretation,
    so ``abs_max`` is reported alongside the signed range: ordering and POD
    analyses use the absolute value.
    """
    rows = []
    groups: List[Any]
    if label is not None and label in df.columns:
        groups = [(str(k), g) for k, g in df.groupby(label, sort=True)]
    else:
        groups = [("(all)", df)]
    for name, g in groups:
        v = pd.to_numeric(g[intensity], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({
            "class": name,
            "n": int(v.size),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "abs_max": float(np.max(np.abs(v))),
            "n_unique": int(np.unique(v).size),
            "mode_guess": guess_intensity_mode(v),
        })
    return pd.DataFrame(rows)
