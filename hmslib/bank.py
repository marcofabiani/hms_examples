"""One model per operating point.

The operating point is known at test time, so there is no need for a single
model covering a multimodal nominal distribution: each OP gets its own detector,
fitted on its own nominal cloud and on its own sensor set (some sensors may be
disabled at some operating points).  :class:`ModelBank` keeps them together and
routes calls by OP name.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .detect.base import BaseDetector

__all__ = ["ModelBank"]

ArrayLike = Union[np.ndarray, pd.DataFrame]


class ModelBank:
    """A detector per operating point, fitted from a :class:`hmslib.io.Dataset`."""

    def __init__(self, models: Optional[Dict[str, BaseDetector]] = None) -> None:
        self.models: Dict[str, BaseDetector] = dict(models or {})

    # -- construction -----------------------------------------------------
    @classmethod
    def fit(
        cls,
        dataset: Any,
        model: BaseDetector,
        operating_points: Optional[Sequence[str]] = None,
        sensors: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> "ModelBank":
        """Deep-copy ``model`` for every operating point and fit it on its nominal data.

        Parameters
        ----------
        model : an *unfitted* detector, used as a template.
        sensors : force a common sensor list (e.g. ``dataset.common_sensors()``)
            instead of each OP's own list.  Needed only if you intend to compare
            scores across operating points.
        """
        names = list(operating_points) if operating_points is not None else \
            list(dataset.operating_points)
        models: Dict[str, BaseDetector] = {}
        for name in names:
            op = dataset[name]
            cols = list(sensors) if sensors is not None else op.sensors
            missing = [c for c in cols if c not in op.nominal.columns]
            if missing:
                raise KeyError(
                    "operating point %r lacks sensor(s) %s; pass sensors="
                    "dataset.common_sensors() to work on the shared subset"
                    % (name, missing)
                )
            det = copy.deepcopy(model)
            det.fit(op.nominal[cols])
            models[name] = det
            if verbose:
                print("fitted %r on %d nominal samples, %d sensors used, "
                      "threshold = %.4g"
                      % (name, len(op.nominal), len(det.features_used_), det.threshold_))
                for w in det.warnings_:
                    print("    ! " + w)
        return cls(models)

    # -- access -----------------------------------------------------------
    def __getitem__(self, name: str) -> BaseDetector:
        if name not in self.models:
            raise KeyError(
                "no model for operating point %r; available: %s"
                % (name, list(self.models))
            )
        return self.models[name]

    def __setitem__(self, name: str, model: BaseDetector) -> None:
        self.models[name] = model

    def __contains__(self, name: str) -> bool:
        return name in self.models

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[str]:
        return iter(self.models)

    @property
    def operating_points(self) -> List[str]:
        return list(self.models)

    # -- delegation -------------------------------------------------------
    def score(self, X: ArrayLike, op: str) -> np.ndarray:
        return self[op].score(X)

    def predict(self, X: ArrayLike, op: str) -> np.ndarray:
        return self[op].predict(X)

    def contributions(self, X: ArrayLike, op: str, **kwargs: Any) -> Any:
        model = self[op]
        if not hasattr(model, "contributions"):
            raise AttributeError(
                "%s does not provide a per sensor decomposition" % type(model).__name__
            )
        return model.contributions(X, **kwargs)

    def score_dataset(self, dataset: Any) -> pd.DataFrame:
        """Score every operating point of a dataset, nominal and failures alike.

        Returns a long table with columns ``op``, ``set`` (``'nominal'`` or
        ``'failure'``), ``score``, ``flagged``, plus ``class`` and ``intensity``
        when available - the shape the evaluation and POD code expects.
        """
        frames = []
        for name, op in dataset.items():
            if name not in self.models:
                continue
            model = self[name]
            cols = model.features_used_
            s_nom = model.score(op.nominal[cols])
            frames.append(pd.DataFrame({
                "op": name,
                "set": "nominal",
                "class": "NOMINAL",
                "intensity": np.nan,
                "score": s_nom,
                "flagged": s_nom > model.threshold_,
            }))
            if len(op.failures):
                s_fail = model.score(op.failures[cols])
                labels = (op.failures[op.label].astype(str).to_numpy()
                          if op.label else np.array(["(unlabelled)"] * len(op.failures)))
                frames.append(pd.DataFrame({
                    "op": name,
                    "set": "failure",
                    "class": labels,
                    "intensity": op.intensity_values(),
                    "score": s_fail,
                    "flagged": s_fail > model.threshold_,
                }))
        if not frames:
            return pd.DataFrame(
                columns=["op", "set", "class", "intensity", "score", "flagged"]
            )
        return pd.concat(frames, ignore_index=True)

    # -- persistence ------------------------------------------------------
    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        index = {}
        for name, model in self.models.items():
            fname = _safe_name(name) + ".joblib"
            model.save(os.path.join(folder, fname))
            index[name] = fname
        with open(os.path.join(folder, "bank.json"), "w", encoding="utf-8") as fh:
            json.dump({"models": index}, fh, indent=2, ensure_ascii=False)
        return folder

    @classmethod
    def load(cls, folder: str) -> "ModelBank":
        with open(os.path.join(folder, "bank.json"), "r", encoding="utf-8") as fh:
            index = json.load(fh)["models"]
        models = {
            name: BaseDetector.load(os.path.join(folder, fname))
            for name, fname in index.items()
        }
        return cls(models)

    # -- reporting --------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """One row per operating point with the key fitted quantities."""
        rows = []
        for name, model in self.models.items():
            d = model.diagnostics_
            rows.append({
                "op": name,
                "model": type(model).__name__,
                "n_train": d.get("n_train"),
                "sensors_used": len(model.features_used_),
                "effective_rank": d.get("effective_rank"),
                "cond": d.get("condition_number"),
                "rule": d.get("threshold_rule"),
                "threshold": model.threshold_,
                "warnings": len(model.warnings_),
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = ["ModelBank: %d operating point(s)" % len(self.models)]
        frame = self.to_frame()
        if len(frame):
            lines.append(frame.to_string(index=False))
        for name, model in self.models.items():
            for w in model.warnings_:
                lines.append("  ! [%s] %s" % (name, w))
        return "\n".join(lines)

    def describe(self) -> None:
        print(self.summary())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ModelBank(%s)" % ", ".join(self.operating_points)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
