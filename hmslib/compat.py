"""Version compatibility layer.

The library is meant to run on an offline machine whose Python and package
versions are not known in advance.  Every API that has been renamed or removed
in recent releases is wrapped here, so the rest of the code never touches it
directly.

Call :func:`check_env` first thing in a session: it reports the versions that
were actually found and raises clear warnings instead of obscure tracebacks.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "check_env",
    "env_info",
    "version_tuple",
    "get_cmap",
    "trapezoid",
    "torch_load",
    "HAS_TORCH",
]


# --------------------------------------------------------------------------
# version helpers
# --------------------------------------------------------------------------

def version_tuple(version: str, n: int = 3) -> Tuple[int, ...]:
    """Parse ``'2.3.5.post1'`` into ``(2, 3, 5)``, ignoring non numeric parts."""
    out: List[int] = []
    for chunk in str(version).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
        if len(out) == n:
            break
    while len(out) < n:
        out.append(0)
    return tuple(out)


def _safe_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


# Minimum versions the code was written against.  Below these, some call sites
# may not exist; above them we rely only on stable APIs.
_MINIMUM = {
    "python": (3, 9, 0),
    "numpy": (1, 20, 0),
    "scipy": (1, 6, 0),
    "pandas": (1, 2, 0),
    "sklearn": (1, 0, 0),
    "matplotlib": (3, 4, 0),
    "torch": (1, 10, 0),
}


def env_info() -> Dict[str, Optional[str]]:
    """Return the detected version of every package the library may use."""
    info: Dict[str, Optional[str]] = {
        "python": "%d.%d.%d" % sys.version_info[:3],
        "platform": sys.platform,
    }
    for name in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "joblib", "torch"):
        info[name] = _safe_version(name)
    return info


def check_env(verbose: bool = True, strict: bool = False) -> Dict[str, Optional[str]]:
    """Report the runtime environment and flag anything that could break.

    Parameters
    ----------
    verbose : print a readable table.
    strict  : raise ``RuntimeError`` instead of warning when a required package
              is missing or too old.

    Returns
    -------
    dict mapping package name to detected version (``None`` if absent).
    """
    info = env_info()
    problems: List[str] = []

    required = ("numpy", "scipy", "pandas", "sklearn", "matplotlib")
    for name in required:
        if info.get(name) is None:
            problems.append("%s is MISSING (required)" % name)

    for name, minimum in _MINIMUM.items():
        got = info.get(name)
        if got in (None, "unknown"):
            continue
        if version_tuple(got) < minimum:
            problems.append(
                "%s %s is older than the tested minimum %s"
                % (name, got, ".".join(str(v) for v in minimum))
            )

    if info.get("torch") is None:
        problems.append("torch is MISSING (only the hmslib.nn subpackage needs it)")

    if verbose:
        print("hmslib environment")
        print("-" * 46)
        for key, value in info.items():
            print("  %-12s %s" % (key, value if value is not None else "-- not found --"))
        print("-" * 46)
        if problems:
            for p in problems:
                print("  ! " + p)
        else:
            print("  no issues detected")

    if problems and strict:
        raise RuntimeError("; ".join(problems))
    for p in problems:
        if "MISSING (required)" in p or "older than" in p:
            warnings.warn(p, RuntimeWarning, stacklevel=2)

    return info


# --------------------------------------------------------------------------
# unstable APIs
# --------------------------------------------------------------------------

def get_cmap(name: str) -> Any:
    """Colormap lookup that works before and after matplotlib 3.9.

    ``plt.cm.get_cmap`` was removed in 3.9; ``matplotlib.colormaps`` does not
    exist before 3.5.
    """
    import matplotlib

    try:
        return matplotlib.colormaps[name]  # matplotlib >= 3.5
    except Exception:
        import matplotlib.cm as cm

        return cm.get_cmap(name)  # matplotlib < 3.9


def categorical_colors(n: int, cmaps: Tuple[str, ...] = ("tab10", "tab20", "tab20b")) -> List[Any]:
    """Return ``n`` visually distinct RGBA colors, cycling over several maps."""
    pool: List[Any] = []
    for name in cmaps:
        cmap = get_cmap(name)
        k = getattr(cmap, "N", 20)
        pool.extend([cmap(i % k) for i in range(k)])
        if len(pool) >= n:
            break
    if len(pool) < n:  # pathological, repeat rather than fail
        pool = (pool * (n // max(len(pool), 1) + 1))
    return pool[:n]


def trapezoid(y: np.ndarray, x: Optional[np.ndarray] = None, **kwargs: Any) -> float:
    """``np.trapezoid`` (numpy >= 2) / ``np.trapz`` (numpy < 2)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x=x, **kwargs) if x is not None else fn(y, **kwargs))


def torch_load(path: str, map_location: str = "cpu") -> Any:
    """``torch.load`` across the ``weights_only`` default change of torch 2.6."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


try:  # pragma: no cover - depends on the machine
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
