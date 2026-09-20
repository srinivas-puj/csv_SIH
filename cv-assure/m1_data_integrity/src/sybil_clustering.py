"""Fake-identity clustering (re-exported from module root)."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_CV_ASSURE = Path(__file__).resolve().parents[2]
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sybil_clustering import cluster_sybils  # noqa: E402

__all__ = ["cluster_sybils"]
