"""Anomaly detectors sharing the contract defined in :mod:`hmslib.detect.base`."""

from __future__ import annotations

from .base import BaseDetector
from .mahalanobis import Mahalanobis

__all__ = ["BaseDetector", "Mahalanobis"]
