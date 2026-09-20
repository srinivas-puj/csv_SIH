"""Tests for the M1 dual-encoder collusion check."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV_ASSURE))
sys.path.insert(0, str(ROOT))

from dual_encoder import compute_encoder_divergence, extract_dual_embeddings  # noqa: E402
from schema import FindingType  # noqa: E402


class _FakeBinding:
    def __init__(self, access_tier: str, with_sub: bool = True) -> None:
        self.access_tier = access_tier
        self.e_sub = object() if with_sub else None
        self.calls: list[Path] = []

    def extract_features(self, image_path: Path):
        self.calls.append(Path(image_path))
        ref = np.arange(8, dtype=np.float32) + Path(image_path).stat().st_size % 3
        sub = np.ones(8, dtype=np.float32) if self.e_sub is not None else None
        return {"e_ref": ref, "e_sub": sub}


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (12, 24, 36)).save(path)
    return path


def test_extract_dual_embeddings_includes_sub_at_t2(tmp_path: Path):
    a = _png(tmp_path / "a.png")
    b = _png(tmp_path / "b.png")
    records = [
        SimpleNamespace(asset_id="a1", file_path=a, contributor_id="c1"),
        SimpleNamespace(asset_id="a2", file_path=b, contributor_id="c1"),
    ]
    binding = _FakeBinding("T2_WHITE_BOX")
    out = extract_dual_embeddings(records, binding)
    assert set(out) == {"a1", "a2"}
    assert out["a1"]["ref_emb"].shape == (8,)
    assert out["a1"]["sub_emb"] is not None
    assert out["a1"]["sub_emb"].shape == (8,)


def test_extract_dual_embeddings_omits_sub_at_t0(tmp_path: Path):
    img = _png(tmp_path / "only.png")
    records = [SimpleNamespace(asset_id="x", file_path=img, contributor_id="c")]
    binding = _FakeBinding("T0_BLACK_BOX", with_sub=True)
    out = extract_dual_embeddings(records, binding)
    assert out["x"]["ref_emb"] is not None
    assert out["x"]["sub_emb"] is None


def test_divergence_flags_high_ref_low_sub():
    finding = compute_encoder_divergence("img-9", ref_dist=5.0, sub_dist=0.4, threshold=2.5, contributor_id="lab")
    assert finding is not None
    assert finding.finding_type is FindingType.DATA_MODEL_COLLUSION
    assert finding.asset_id == "img-9"
    assert finding.contributor_id == "lab"
    assert 0.0 < finding.score <= 1.0
    assert finding.details["confidence"] >= 0.55
    assert finding.details["divergence"] == pytest.approx(5.0 / (0.4 + 1e-6))


def test_divergence_none_when_both_ordinary():
    assert compute_encoder_divergence("img-0", ref_dist=0.8, sub_dist=0.7, threshold=2.5) is None


def test_divergence_none_when_both_outliers():
    # Ratio ~1 even if both distances are large.
    assert compute_encoder_divergence("img-1", ref_dist=6.0, sub_dist=5.5, threshold=2.5) is None
