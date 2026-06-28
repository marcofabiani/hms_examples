"""
Helper functions for the SSME health monitoring example.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis


SENSOR_NAMES = [
    'Pc', 'HPFTP_w', 'HPOTP_w', 'LPFTP_w', 'LPOTP_w', 'PBOBP_w',
    'Pc_FPB', 'Pc_OPB', 'MFV_m', 'MOV_m', 'OPOV_m', 'FPOV_m',
    'T_Nozzle_Cooling_exit', 'HGM_Temp', 'Man_OX_Temp', 'Man_H2_Temp',
    'Man_MCC_COOL_Pres', 'HPFTP_p_pump_out', 'HPOTP_p_pump_out',
    'LPFTP_p_pump_out', 'LPOTP_p_pump_out', 'PBOBP_p_pump_out',
    'HPFTP_T_turb_inlet', 'HPOTP_T_turb_inlet',
    'LPFTP_T_turb_inlet', 'LPOTP_T_turb_inlet',
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nominal(path: str) -> pd.DataFrame:
    """Load the nominal operating data CSV.

    Returns a DataFrame with 26 sensor columns and no NaN/inf rows.
    """
    df = pd.read_csv(path)
    df = df[SENSOR_NAMES]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    return df


def load_failures(path: str) -> pd.DataFrame:
    """Load the failure data CSV.

    Returns a DataFrame with 26 sensor columns plus a 'Failure' label column.
    """
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(df_nominal: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on nominal data and return it."""
    scaler = StandardScaler()
    scaler.fit(df_nominal[SENSOR_NAMES])
    return scaler


def scale(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform a DataFrame (must have SENSOR_NAMES columns) using a fitted scaler."""
    return scaler.transform(df[SENSOR_NAMES])


# ---------------------------------------------------------------------------
# Mahalanobis
# ---------------------------------------------------------------------------

def fit_mahalanobis(X_train: np.ndarray):
    """Estimate mean and precision matrix from training data using LedoitWolf.

    LedoitWolf gives a better-conditioned covariance estimate than the
    sample covariance when the number of features is not negligible compared
    to the number of samples.

    Returns
    -------
    mu : ndarray, shape (n_features,)
    precision : ndarray, shape (n_features, n_features)  — inverse covariance
    """
    mu = X_train.mean(axis=0)
    lw = LedoitWolf().fit(X_train)
    return mu, lw.precision_


def mahalanobis_distances(X: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Compute the Mahalanobis distance of each row in X from mu."""
    return np.array([mahalanobis(x, mu, precision) for x in X])


def detection_threshold(distances_train: np.ndarray, percentile: float = 99.0) -> float:
    """Return a detection threshold as a high percentile of training distances."""
    return float(np.percentile(distances_train, percentile))


# ---------------------------------------------------------------------------
# Scatter / ellipse utilities
# ---------------------------------------------------------------------------

def ellipse_points(mean: np.ndarray, cov: np.ndarray,
                   radius: float = 1.0, n_points: int = 300):
    """Return (x, y) arrays tracing the sigma-ellipse of a 2-D Gaussian.

    Parameters
    ----------
    mean   : (2,) centre of the ellipse
    cov    : (2, 2) covariance matrix
    radius : number of sigma (e.g. 3 → 3σ contour)
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    transform = evecs @ np.diag(np.sqrt(np.maximum(evals, 0.0))) * radius
    pts = mean.reshape(2, 1) + transform @ circle
    return pts[0], pts[1]


def failure_palette(classes):
    """Return a dict {class_name: rgba_color} for up to 40 classes.

    Uses tab20 for the first 20 entries and tab20b for the rest.
    """
    import matplotlib.cm as cm
    c1 = [cm.tab20(i / 19)  for i in range(20)]
    c2 = [cm.tab20b(i / 19) for i in range(20)]
    pool = (c1 + c2)[:len(classes)]
    return dict(zip(classes, pool))
