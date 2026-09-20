"""k-NN label-agreement detector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detectors import LabelFlipDetector  # noqa: E402

__all__ = ["LabelFlipDetector"]
