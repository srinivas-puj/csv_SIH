"""Activation-clustering detector (re-exported from module-root detectors)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detectors import ActivationClusteringDetector  # noqa: E402

__all__ = ["ActivationClusteringDetector"]
