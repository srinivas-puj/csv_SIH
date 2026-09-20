"""Dual-encoder cross-check engine for Module M1.

Compares a pinned reference encoder (E_ref / DINOv2-S) against the contributed
subject encoder (E_sub). A sample that is an outlier under E_ref but ordinary
under E_sub is treated as DATA_MODEL_COLLUSION: E_sub appears trained to
accommodate poisoned data.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import FindingType, SampleAnomaly  # noqa: E402

logger = logging.getLogger(__name__)

EPS = 1e-6
DEFAULT_DIVERGENCE_THRESHOLD = 2.5

PathLike = Union[str, Path]


def extract_dual_embeddings(
    records: List[Any],
    encoder_binding: Any,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """Extract E_ref (and E_sub when T1/T2) embeddings for every asset.

    Args:
        records: Ingested ``AssetRecord`` objects (need ``asset_id`` and ``file_path``).
        encoder_binding: M0 ``DualEncoderBinding`` (or a compatible stub) exposing
            ``extract_features(image_path) -> {"e_ref"|"ref_emb", "e_sub"|"sub_emb"}``
            and an ``access_tier`` of T0/T1/T2.

    Returns:
        ``{asset_id: {"ref_emb": ndarray, "sub_emb": Optional[ndarray]}}``.
        ``sub_emb`` is ``None`` when the subject encoder is unavailable (T0).
    """
    if encoder_binding is None:
        raise ValueError("encoder_binding is required")

    subject_ok = _subject_encoder_available(encoder_binding)
    embeddings: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

    for record in records:
        asset_id = str(_record_field(record, "asset_id"))
        image_path = Path(str(_record_field(record, "file_path")))
        try:
            ref_emb, sub_emb = _embed_one(encoder_binding, image_path, subject_ok)
        except Exception as exc:  # noqa: BLE001 - skip corrupt assets, keep the batch going
            logger.warning("Dual-encoder extract failed for %s (%s): %s", asset_id, image_path, exc)
            continue
        embeddings[asset_id] = {"ref_emb": ref_emb, "sub_emb": sub_emb}
    return embeddings


def compute_encoder_divergence(
    asset_id: str,
    ref_dist: float,
    sub_dist: float,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    *,
    contributor_id: str = "unknown",
) -> Optional[SampleAnomaly]:
    """Flag DATA_MODEL_COLLUSION when E_ref sees an outlier that E_sub treats as normal.

    ``ref_dist`` / ``sub_dist`` are anomaly distances in each encoder's feature
    space (e.g. Mahalanobis to the class centroid). Divergence is

        ref_dist / (sub_dist + 1e-6)

    A large ratio means the reference encoder flags the sample while the
    subject encoder embeds it as ordinary — evidence that E_sub was trained
    to accommodate the poison.

    Returns:
        A ``SampleAnomaly`` when ``divergence > threshold``, otherwise ``None``.
    """
    ref = max(0.0, float(ref_dist))
    sub = max(0.0, float(sub_dist))
    divergence = ref / (sub + EPS)

    if divergence <= threshold:
        return None

    # Map excess asymmetry onto (0, 1]; saturates toward 1 as the ratio grows.
    excess = math.log(divergence / threshold)
    score = float(min(1.0, 1.0 - math.exp(-excess)))
    if score <= 0.0:
        return None

    confidence = float(min(0.99, 0.55 + 0.45 * score))
    return SampleAnomaly(
        asset_id=asset_id,
        contributor_id=contributor_id,
        finding_type=FindingType.DATA_MODEL_COLLUSION,
        score=score,
        details={
            "ref_dist": ref,
            "sub_dist": sub,
            "divergence": divergence,
            "threshold": float(threshold),
            "confidence": confidence,
            "interpretation": (
                "E_ref distance high while E_sub distance low: subject encoder "
                "embeds a reference-outlier as in-distribution (possible collusion)."
            ),
        },
    )


def scan_collusion(
    embeddings: Mapping[str, Mapping[str, Optional[np.ndarray]]],
    records: Iterable[Any],
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> List[SampleAnomaly]:
    """Convenience: Mahalanobis-style distances vs embedding cloud, then diverge."""
    contributor_by_id = {
        str(_record_field(record, "asset_id")): str(_record_field(record, "contributor_id", "unknown"))
        for record in records
    }
    ref_cloud = [np.asarray(row["ref_emb"], dtype=np.float64).reshape(-1) for row in embeddings.values() if row.get("ref_emb") is not None]
    sub_cloud = [np.asarray(row["sub_emb"], dtype=np.float64).reshape(-1) for row in embeddings.values() if row.get("sub_emb") is not None]
    if not ref_cloud:
        return []

    findings: List[SampleAnomaly] = []
    for asset_id, row in embeddings.items():
        ref_emb = row.get("ref_emb")
        sub_emb = row.get("sub_emb")
        if ref_emb is None or sub_emb is None:
            continue
        ref_dist = _distance_to_cloud(np.asarray(ref_emb, dtype=np.float64).reshape(-1), ref_cloud)
        sub_dist = _distance_to_cloud(np.asarray(sub_emb, dtype=np.float64).reshape(-1), sub_cloud)
        hit = compute_encoder_divergence(
            asset_id,
            ref_dist,
            sub_dist,
            threshold,
            contributor_id=contributor_by_id.get(asset_id, "unknown"),
        )
        if hit is not None:
            findings.append(hit)
    return findings


def _subject_encoder_available(encoder_binding: Any) -> bool:
    """True when the binding has an E_sub and access is T1 or T2."""
    if getattr(encoder_binding, "e_sub", None) is None and not _has_sub_extract(encoder_binding):
        # Still allow T1/T2 if extract_features itself returns e_sub.
        pass
    rank = _tier_rank(getattr(encoder_binding, "access_tier", None))
    if rank >= 1:
        return True
    # Explicit e_sub with no tier declared: treat as available.
    if getattr(encoder_binding, "access_tier", None) is None and getattr(encoder_binding, "e_sub", None) is not None:
        return True
    return False


def _has_sub_extract(encoder_binding: Any) -> bool:
    return callable(getattr(encoder_binding, "extract_features", None))


def _tier_rank(tier: Any) -> int:
    if tier is None:
        return 0
    rank = getattr(tier, "rank", None)
    if isinstance(rank, (int, float)):
        return int(rank)
    text = str(getattr(tier, "value", tier)).upper()
    if "T2" in text or "WEIGHT" in text or "WHITE" in text:
        return 2
    if "T1" in text or "LOGIT" in text or "GREY" in text or "GRAY" in text:
        return 1
    return 0


def _embed_one(encoder_binding: Any, image_path: Path, subject_ok: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if not callable(getattr(encoder_binding, "extract_features", None)):
        raise TypeError("encoder_binding must implement extract_features(image_path)")

    payload = encoder_binding.extract_features(image_path)
    if not isinstance(payload, Mapping):
        raise TypeError("extract_features must return a mapping with e_ref / e_sub")

    ref_raw = payload.get("e_ref", payload.get("ref_emb"))
    if ref_raw is None:
        raise ValueError(f"E_ref embedding missing for {image_path}")
    ref_emb = np.asarray(ref_raw, dtype=np.float32).reshape(-1)

    sub_emb: Optional[np.ndarray] = None
    if subject_ok:
        sub_raw = payload.get("e_sub", payload.get("sub_emb"))
        if sub_raw is not None:
            sub_emb = np.asarray(sub_raw, dtype=np.float32).reshape(-1)
    return ref_emb, sub_emb


def _record_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _distance_to_cloud(vector: np.ndarray, cloud: List[np.ndarray]) -> float:
    stacked = np.stack(cloud, axis=0)
    centroid = stacked.mean(axis=0)
    delta = vector - centroid
    # Diagonal Mahalanobis fallback: variance-normalized Euclidean.
    var = stacked.var(axis=0) + EPS
    return float(np.sqrt(np.sum((delta * delta) / var)))
