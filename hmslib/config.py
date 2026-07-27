"""Global configuration: random seed and figure style."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["PlotConfig", "PLOT", "apply_style", "set_seed", "get_rng", "DEFAULT_SEED"]


DEFAULT_SEED = 0


@dataclass
class PlotConfig:
    """Figure defaults, applied by :func:`apply_style`."""

    dpi: int = 110
    savefig_dpi: int = 200
    font_size: float = 9.0
    title_size: float = 10.0
    figsize: Tuple[float, float] = (7.0, 4.5)
    grid_alpha: float = 0.25
    scatter_size: float = 8.0
    scatter_alpha: float = 0.45
    nominal_color: str = "#8c8c8c"
    threshold_color: str = "#c81e1e"
    max_scatter_points: int = 4000  # subsample above this, for readable PDFs
    extra_rc: Dict[str, Any] = field(default_factory=dict)


PLOT = PlotConfig()


def apply_style(cfg: Optional[PlotConfig] = None) -> None:
    """Apply the library's matplotlib defaults.

    Only stable rcParams keys are touched, so it works on any matplotlib >= 3.4.
    """
    import matplotlib as mpl

    cfg = cfg or PLOT
    rc = {
        "figure.dpi": cfg.dpi,
        "savefig.dpi": cfg.savefig_dpi,
        "figure.figsize": cfg.figsize,
        "font.size": cfg.font_size,
        "axes.titlesize": cfg.title_size,
        "axes.labelsize": cfg.font_size,
        "axes.grid": True,
        "grid.alpha": cfg.grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": cfg.font_size - 1,
        "xtick.labelsize": cfg.font_size - 1,
        "ytick.labelsize": cfg.font_size - 1,
        "savefig.bbox": "tight",
    }
    rc.update(cfg.extra_rc)
    mpl.rcParams.update(rc)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python, numpy and (if present) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - depends on the machine
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def get_rng(seed: Optional[int] = DEFAULT_SEED) -> np.random.RandomState:
    """Return a legacy ``RandomState``.

    Deliberately not ``default_rng``: ``RandomState`` is what scikit-learn
    accepts as ``random_state`` on every version, so the same object can be
    handed to both our code and sklearn's.
    """
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)
