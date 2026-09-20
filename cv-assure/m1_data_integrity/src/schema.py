"""M1 schemas (re-exported from the module-root ``schema.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_CV_ASSURE = Path(__file__).resolve().parents[2]
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import (  # noqa: E402
    CapabilityMatrix,
    ContributorRollup,
    FindingType,
    M1Result,
    SampleAnomaly,
)

__all__ = [
    "CapabilityMatrix",
    "ContributorRollup",
    "FindingType",
    "M1Result",
    "SampleAnomaly",
]
