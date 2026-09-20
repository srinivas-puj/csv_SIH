"""M1 sample-level integrity detectors (CV-ASSURE v2, Clause 2.2.1).

Each detector subclasses the M0 ``BaseDetector`` contract: ``check_availability``
gates on the capability matrix, and ``run`` returns an ``UNAVAILABLE`` payload
instead of raising when prerequisites are missing.
"""

from __future__ import annotations

import logging
import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import FindingType, SampleAnomaly  # noqa: E402

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_ASPECT_RATIO = 20.0
PHASH_HAMMING_MAX = 8
COSINE_DUP_MIN = 0.97
LABEL_FLIP_K = 5
LABEL_FLIP_AGREEMENT = 0.4
OOD_PERCENTILE = 99.0
SPECTRAL_ZSCORE = 3.0


def _try_import_m0_base() -> Optional[type]:
    try:
        from m0_ingest.base_detector import BaseDetector as _Base  # type: ignore

        return _Base
    except ImportError:
        try:
            from base_detector import BaseDetector as _Base  # type: ignore

            return _Base
        except ImportError:
            return None


class _V2BaseDetector(ABC):
    """M0-compatible capability-gated detector (used when M0 v2 is not importable)."""

    def __init__(
        self,
        detector_id: str,
        required_tier: Any,
        supported_tasks: Sequence[Any],
        requires_reference: bool = False,
    ) -> None:
        self.detector_id = detector_id
        self.required_tier = required_tier
        self.supported_tasks = list(supported_tasks)
        self.requires_reference = requires_reference
        # M0 v1 aliases
        self.name = detector_id
        self.min_required_tier = required_tier

    def check_availability(self, capability_matrix: Any) -> Tuple[bool, str]:
        task = _normalize_task(_matrix_field(capability_matrix, "task_type"))
        supported = {_normalize_task(item) for item in self.supported_tasks}
        if task not in supported:
            return False, f"{self.detector_id.upper()}_UNAVAILABLE_FOR_TASK_TYPE"

        have = _tier_rank(_matrix_field(capability_matrix, "access_tier"))
        need = _tier_rank(self.required_tier)
        if have < need:
            return (
                False,
                (
                    "UNAVAILABLE_AT_TIER: "
                    f"requires {_tier_name(self.required_tier)}, "
                    f"found {_tier_name(_matrix_field(capability_matrix, 'access_tier'))}"
                ),
            )

        if self.requires_reference and _is_unreferenced(capability_matrix):
            return False, f"{self.detector_id.upper()}_UNAVAILABLE_WITHOUT_REFERENCE"

        return True, f"SUPPORTED: detector={self.detector_id}"

    def run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> Dict[str, Any]:
        allowed, reason = self.check_availability(capability_matrix)
        if not allowed:
            return {
                "detector_id": self.detector_id,
                "status": "UNAVAILABLE",
                "reason": reason,
                "score": None,
                "anomalies": [],
            }
        anomalies = self._run(dataset, capability_matrix, **kwargs)
        scores = [float(item.score) for item in anomalies]
        return {
            "detector_id": self.detector_id,
            "status": "OK",
            "reason": None,
            "score": max(scores) if scores else 0.0,
            "anomalies": anomalies,
        }

    @abstractmethod
    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        raise NotImplementedError


_ImportedBase = _try_import_m0_base()
BaseDetector = _ImportedBase or _V2BaseDetector  # type: ignore[misc, assignment]


def _matrix_field(matrix: Any, name: str, default: Any = None) -> Any:
    if isinstance(matrix, Mapping):
        return matrix.get(name, default)
    return getattr(matrix, name, default)


def _normalize_task(task: Any) -> str:
    text = str(getattr(task, "value", task) or "").upper()
    if "SEG" in text:
        return "SEGMENTATION"
    if "DET" in text or "BBOX" in text or "VISION" in text:
        return "DETECTION"
    if "CLASS" in text:
        return "CLASSIFICATION"
    return text or "DETECTION"


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


def _tier_name(tier: Any) -> str:
    if hasattr(tier, "name") and isinstance(tier.name, str) and tier.name:
        return tier.name
    return str(getattr(tier, "value", tier))


def _is_unreferenced(matrix: Any) -> bool:
    mode = str(_matrix_field(matrix, "reference_mode") or "").upper()
    digest = _matrix_field(matrix, "reference_profile_digest")
    return mode in {"UNREFERENCED", ""} and not digest


def _record_get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _asset_id(record: Any) -> str:
    return str(_record_get(record, "asset_id") or "unknown")


def _contributor_id(record: Any) -> str:
    return str(_record_get(record, "contributor_id") or "unknown")


def _annotations(record: Any) -> List[Any]:
    raw = _record_get(record, "annotations") or []
    return list(raw)


def _image_size(record: Any) -> Tuple[Optional[float], Optional[float]]:
    meta = _record_get(record, "image_metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {
            "width": getattr(meta, "width", None),
            "height": getattr(meta, "height", None),
        }
    width = meta.get("width")
    height = meta.get("height")
    try:
        return (float(width) if width is not None else None, float(height) if height is not None else None)
    except (TypeError, ValueError):
        return None, None


def _bbox(annotation: Any) -> Optional[List[float]]:
    raw = annotation.get("bbox") if isinstance(annotation, Mapping) else getattr(annotation, "bbox", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(part) for part in raw]
    except (TypeError, ValueError):
        return None


def _class_name(annotation: Any, record: Any = None) -> str:
    if isinstance(annotation, Mapping):
        name = annotation.get("class_name") or annotation.get("label")
    else:
        name = getattr(annotation, "class_name", None) or getattr(annotation, "label", None)
    if name:
        return str(name)
    if record is not None:
        fallback = _record_get(record, "class_name") or _record_get(record, "label")
        if fallback:
            return str(fallback)
    return "unknown"


def _primary_label(record: Any) -> str:
    anns = _annotations(record)
    if not anns:
        return str(_record_get(record, "class_name") or _record_get(record, "label") or "unknown")
    return _class_name(anns[0], record)


def _ref_embedding(embeddings: Mapping[str, Any], asset_id: str) -> Optional[np.ndarray]:
    row = embeddings.get(asset_id)
    if row is None:
        return None
    if isinstance(row, Mapping):
        raw = row.get("ref_emb", row.get("e_ref"))
    else:
        raw = row
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# 1. Annotation geometry
# ---------------------------------------------------------------------------


class AnnotationAnomalyDetector(BaseDetector):
    """Flag impossible boxes: aspect, area, canvas, and class-size outliers."""

    def __init__(self, max_aspect_ratio: float = MAX_ASPECT_RATIO, size_z: float = 4.0) -> None:
        super().__init__(
            detector_id="annotation_anomaly_detector",
            required_tier="T0",
            supported_tasks=["DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.max_aspect_ratio = max_aspect_ratio
        self.size_z = size_z

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        areas_by_class: Dict[str, List[float]] = defaultdict(list)
        parsed: List[Tuple[Any, Any, List[float], str]] = []
        for record in dataset:
            width, height = _image_size(record)
            for annotation in _annotations(record):
                box = _bbox(annotation)
                if box is None:
                    continue
                _x, _y, w, h = box
                class_name = _class_name(annotation, record)
                parsed.append((record, annotation, box, class_name))
                if w > 0 and h > 0:
                    areas_by_class[class_name].append(w * h)
            _ = (width, height)

        stats = {
            name: (float(np.mean(vals)), float(np.std(vals)) + EPS)
            for name, vals in areas_by_class.items()
            if len(vals) >= 4
        }

        findings: List[SampleAnomaly] = []
        for record, _annotation, box, class_name in parsed:
            reasons = self._box_reasons(box, _image_size(record), class_name, stats)
            if not reasons:
                continue
            findings.append(
                SampleAnomaly(
                    asset_id=_asset_id(record),
                    contributor_id=_contributor_id(record),
                    finding_type=FindingType.ANNOTATION_ANOMALY,
                    score=min(1.0, 0.35 * len(reasons)),
                    details={"bbox": box, "reasons": reasons, "class_name": class_name},
                )
            )
        return findings

    def _box_reasons(
        self,
        box: List[float],
        size: Tuple[Optional[float], Optional[float]],
        class_name: str,
        stats: Mapping[str, Tuple[float, float]],
    ) -> List[str]:
        x, y, w, h = box
        reasons: List[str] = []
        if w <= 0 or h <= 0:
            reasons.append("zero_or_negative_area")
        else:
            aspect = max(w / h, h / w)
            if aspect > self.max_aspect_ratio:
                reasons.append(f"aspect_ratio_{aspect:.1f}")
        if w < 1.0 or h < 1.0:
            reasons.append("sub_pixel_extent")
        width, height = size
        if width is not None and height is not None:
            if x + w <= 0 or y + h <= 0 or x >= width or y >= height:
                reasons.append("off_canvas")
            if x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
                reasons.append("extends_beyond_canvas")
        if class_name in stats and w > 0 and h > 0:
            mean, std = stats[class_name]
            z = abs((w * h) - mean) / std
            if z >= self.size_z:
                reasons.append(f"class_size_outlier_z_{z:.1f}")
        return reasons


# ---------------------------------------------------------------------------
# 2. Near-duplicates
# ---------------------------------------------------------------------------


class NearDuplicateDetector(BaseDetector):
    """pHash / dHash plus E_ref cosine near-duplicate flood detector."""

    def __init__(
        self,
        hamming_max: int = PHASH_HAMMING_MAX,
        cosine_min: float = COSINE_DUP_MIN,
    ) -> None:
        super().__init__(
            detector_id="near_duplicate_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.hamming_max = hamming_max
        self.cosine_min = cosine_min

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        hashes: List[Tuple[str, Any, int, int, Optional[np.ndarray]]] = []
        for record in dataset:
            path = Path(str(_record_get(record, "file_path") or ""))
            phash, dhash = _perceptual_hashes(path)
            emb = _ref_embedding(embeddings, _asset_id(record))
            hashes.append((_asset_id(record), record, phash, dhash, emb))

        flagged: Dict[str, SampleAnomaly] = {}
        for i, (id_a, rec_a, pha, dha, ea) in enumerate(hashes):
            mates: List[Dict[str, Any]] = []
            for id_b, rec_b, phb, dhb, eb in hashes[i + 1 :]:
                ham_p = _hamming(pha, phb) if pha is not None and phb is not None else None
                ham_d = _hamming(dha, dhb) if dha is not None and dhb is not None else None
                cosine = _cosine(ea, eb) if ea is not None and eb is not None else None
                hash_hit = (ham_p is not None and ham_p <= self.hamming_max) or (
                    ham_d is not None and ham_d <= self.hamming_max
                )
                emb_hit = cosine is not None and cosine >= self.cosine_min
                if not (hash_hit or emb_hit):
                    continue
                mates.append(
                    {
                        "other_asset_id": id_b,
                        "other_contributor_id": _contributor_id(rec_b),
                        "phash_hamming": ham_p,
                        "dhash_hamming": ham_d,
                        "ref_cosine": cosine,
                        "cross_contributor": _contributor_id(rec_a) != _contributor_id(rec_b),
                    }
                )
            if not mates:
                continue
            score = min(1.0, 0.4 + 0.15 * len(mates))
            flagged[id_a] = SampleAnomaly(
                asset_id=id_a,
                contributor_id=_contributor_id(rec_a),
                finding_type=FindingType.NEAR_DUPLICATE_FLOOD,
                score=score,
                details={"matches": mates[:16], "match_count": len(mates)},
            )
        return list(flagged.values())


def _perceptual_hashes(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path.is_file():
        return None, None
    try:
        try:
            import imagehash

            with Image.open(path) as image:
                rgb = image.convert("RGB")
                return int(str(imagehash.phash(rgb)), 16), int(str(imagehash.dhash(rgb)), 16)
        except ImportError:
            return _numpy_phash(path), _numpy_dhash(path)
    except (OSError, ValueError) as exc:
        logger.debug("Hash failed for %s: %s", path, exc)
        return None, None


def _numpy_dhash(path: Path, hash_size: int = 8) -> Optional[int]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _numpy_phash(path: Path, hash_size: int = 8) -> Optional[int]:
    from scipy.fftpack import dct

    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size].flatten()
    median = float(np.median(low[1:]))  # skip DC
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count()) if hasattr(int, "bit_count") else bin(a ^ b).count("1")


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return None
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 3. Label flip (k-NN in E_ref)
# ---------------------------------------------------------------------------


class LabelFlipDetector(BaseDetector):
    """Flag samples whose label disagrees with k-NN neighbours in E_ref space."""

    def __init__(self, k: int = LABEL_FLIP_K, min_agreement: float = LABEL_FLIP_AGREEMENT) -> None:
        super().__init__(
            detector_id="label_flip_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.k = k
        self.min_agreement = min_agreement

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        ids: List[str] = []
        labels: List[str] = []
        vectors: List[np.ndarray] = []
        records_by_id: Dict[str, Any] = {}
        for record in dataset:
            asset_id = _asset_id(record)
            emb = _ref_embedding(embeddings, asset_id)
            if emb is None:
                continue
            ids.append(asset_id)
            labels.append(_primary_label(record))
            vectors.append(emb)
            records_by_id[asset_id] = record
        if len(vectors) < max(3, self.k + 1):
            return []

        stacked = np.stack(vectors, axis=0)
        try:
            from sklearn.neighbors import NearestNeighbors

            knn = NearestNeighbors(n_neighbors=min(self.k + 1, len(vectors)), metric="cosine")
            knn.fit(stacked)
            indices = knn.kneighbors(return_distance=False)
        except Exception:  # noqa: BLE001
            indices = _brute_knn(stacked, min(self.k + 1, len(vectors)))

        findings: List[SampleAnomaly] = []
        for row_i, neighbours in enumerate(indices):
            peer_idx = [j for j in neighbours if j != row_i][: self.k]
            if not peer_idx:
                continue
            peer_labels = [labels[j] for j in peer_idx]
            assigned = labels[row_i]
            agree = sum(lab == assigned for lab in peer_labels) / len(peer_labels)
            if agree >= self.min_agreement:
                continue
            majority = max(set(peer_labels), key=peer_labels.count)
            if majority == assigned:
                continue
            findings.append(
                SampleAnomaly(
                    asset_id=ids[row_i],
                    contributor_id=_contributor_id(records_by_id[ids[row_i]]),
                    finding_type=FindingType.LABEL_FLIP,
                    score=float(min(1.0, 1.0 - agree)),
                    details={
                        "assigned_label": assigned,
                        "knn_majority": majority,
                        "knn_agreement": agree,
                        "k": len(peer_idx),
                    },
                )
            )
        return findings


def _brute_knn(stacked: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(stacked, axis=1, keepdims=True) + EPS
    sim = (stacked @ stacked.T) / (norms @ norms.T)
    return np.argsort(-sim, axis=1)[:, :k]


# ---------------------------------------------------------------------------
# 4. OOD insertion
# ---------------------------------------------------------------------------


class OODInsertionDetector(BaseDetector):
    """Class-conditional Mahalanobis distance vs reference (or bootstrapped) stats."""

    def __init__(self, percentile: float = OOD_PERCENTILE) -> None:
        super().__init__(
            detector_id="ood_insertion_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=True,
        )
        self.percentile = percentile

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        profile: Mapping[str, Any] = kwargs.get("reference_profile") or {}
        grouped: Dict[str, List[Tuple[Any, np.ndarray]]] = defaultdict(list)
        for record in dataset:
            emb = _ref_embedding(embeddings, _asset_id(record))
            if emb is None:
                continue
            grouped[_primary_label(record)].append((record, emb))

        stats = _class_stats(grouped, profile)
        findings: List[SampleAnomaly] = []
        for class_name, rows in grouped.items():
            mean_cov = stats.get(class_name)
            if mean_cov is None:
                continue
            mean, precision = mean_cov
            distances = [_mahalanobis(emb, mean, precision) for _record, emb in rows]
            if len(distances) >= 5:
                cutoff = float(np.percentile(distances, self.percentile))
            else:
                cutoff = float(np.mean(distances) + 3.0 * (np.std(distances) + EPS))
            for (record, _emb), dist in zip(rows, distances):
                if dist < cutoff:
                    continue
                span = max(dist, cutoff, EPS)
                findings.append(
                    SampleAnomaly(
                        asset_id=_asset_id(record),
                        contributor_id=_contributor_id(record),
                        finding_type=FindingType.OOD_INSERTION,
                        score=float(min(1.0, (dist - cutoff) / span + 0.5)),
                        details={
                            "class_name": class_name,
                            "mahalanobis": dist,
                            "percentile_cutoff": cutoff,
                            "percentile": self.percentile,
                        },
                    )
                )
        return findings


def _class_stats(
    grouped: Mapping[str, List[Tuple[Any, np.ndarray]]],
    profile: Mapping[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    profile_classes = profile.get("classes") if isinstance(profile.get("classes"), Mapping) else profile
    for class_name, rows in grouped.items():
        prior = None
        if isinstance(profile_classes, Mapping):
            prior = profile_classes.get(class_name)
        if isinstance(prior, Mapping) and "mean" in prior:
            mean = np.asarray(prior["mean"], dtype=np.float64).reshape(-1)
            cov = np.asarray(prior.get("cov") or prior.get("covariance") or np.eye(mean.size), dtype=np.float64)
            if cov.ndim == 1:
                cov = np.diag(cov)
            precision = np.linalg.pinv(cov + EPS * np.eye(cov.shape[0]))
            stats[class_name] = (mean, precision)
            continue
        if len(rows) < 2:
            continue
        stacked = np.stack([emb for _r, emb in rows], axis=0)
        mean = stacked.mean(axis=0)
        cov = np.cov(stacked, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]])
        precision = np.linalg.pinv(cov + EPS * np.eye(cov.shape[0]))
        stats[class_name] = (mean, precision)
    return stats


def _mahalanobis(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    delta = x.reshape(-1) - mean.reshape(-1)
    dim = min(delta.size, precision.shape[0])
    delta = delta[:dim]
    prec = precision[:dim, :dim]
    return float(math.sqrt(max(0.0, delta @ prec @ delta)))


# ---------------------------------------------------------------------------
# 5. Spectral signatures (Tran et al.)
# ---------------------------------------------------------------------------


class SpectralSignatureDetector(BaseDetector):
    """Flag high projections onto the top class-conditional singular vector."""

    def __init__(self, z_threshold: float = SPECTRAL_ZSCORE) -> None:
        super().__init__(
            detector_id="spectral_signature_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.z_threshold = z_threshold

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        grouped: Dict[str, List[Tuple[Any, np.ndarray]]] = defaultdict(list)
        for record in dataset:
            emb = _ref_embedding(embeddings, _asset_id(record))
            if emb is None:
                continue
            grouped[_primary_label(record)].append((record, emb))

        findings: List[SampleAnomaly] = []
        for class_name, rows in grouped.items():
            if len(rows) < 4:
                continue
            stacked = np.stack([emb for _r, emb in rows], axis=0)
            centered = stacked - stacked.mean(axis=0, keepdims=True)
            try:
                _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            direction = vt[0]
            projections = centered @ direction
            mu = float(np.mean(projections))
            sd = float(np.std(projections)) + EPS
            for (record, _emb), proj in zip(rows, projections):
                z = abs(float(proj) - mu) / sd
                if z < self.z_threshold:
                    continue
                findings.append(
                    SampleAnomaly(
                        asset_id=_asset_id(record),
                        contributor_id=_contributor_id(record),
                        finding_type=FindingType.TRIGGER_INJECTION,
                        score=float(min(1.0, z / (self.z_threshold * 2.0))),
                        details={
                            "class_name": class_name,
                            "spectral_projection": float(proj),
                            "z_score": z,
                            "z_threshold": self.z_threshold,
                        },
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# 6. Activation clustering (Chen et al.)
# ---------------------------------------------------------------------------


class ActivationClusteringDetector(BaseDetector):
    """2-means on E_ref activations; the minority cluster is treated as a trigger set."""

    def __init__(self, max_minority_fraction: float = 0.4, min_separation: float = 2.0) -> None:
        super().__init__(
            detector_id="activation_clustering_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.max_minority_fraction = max_minority_fraction
        self.min_separation = min_separation

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        grouped: Dict[str, List[Tuple[Any, np.ndarray]]] = defaultdict(list)
        for record in dataset:
            emb = _ref_embedding(embeddings, _asset_id(record))
            if emb is None:
                continue
            grouped[_primary_label(record)].append((record, emb))

        findings: List[SampleAnomaly] = []
        for class_name, rows in grouped.items():
            if len(rows) < 8:
                continue
            stacked = np.stack([emb for _r, emb in rows], axis=0)
            labels, centers = _two_means(stacked)
            if labels is None or centers is None:
                continue
            counts = np.bincount(labels, minlength=2)
            minority = int(np.argmin(counts))
            n = int(labels.size)
            minority_n = int(counts[minority])
            if minority_n == 0 or minority_n / n > self.max_minority_fraction:
                continue
            sep = float(np.linalg.norm(centers[0] - centers[1]) / (np.std(stacked) + EPS))
            if sep < self.min_separation:
                continue
            score = float(min(1.0, 0.45 + 0.15 * sep + 0.2 * (1.0 - minority_n / n)))
            for (record, _emb), cluster in zip(rows, labels):
                if int(cluster) != minority:
                    continue
                findings.append(
                    SampleAnomaly(
                        asset_id=_asset_id(record),
                        contributor_id=_contributor_id(record),
                        finding_type=FindingType.TRIGGER_INJECTION,
                        score=score,
                        details={
                            "class_name": class_name,
                            "cluster": int(cluster),
                            "minority_size": minority_n,
                            "class_size": n,
                            "centroid_separation": sep,
                            "method": "activation_clustering",
                        },
                    )
                )
        return findings


def _two_means(stacked: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = model.fit_predict(stacked)
        return np.asarray(labels, dtype=np.int64), np.asarray(model.cluster_centers_, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # Lloyd fallback with two farthest points as seeds.
        diffs = stacked - stacked.mean(axis=0, keepdims=True)
        seed_a = int(np.argmax(np.linalg.norm(diffs, axis=1)))
        seed_b = int(np.argmax(np.linalg.norm(stacked - stacked[seed_a], axis=1)))
        centers = np.stack([stacked[seed_a], stacked[seed_b]], axis=0)
        labels = np.zeros(len(stacked), dtype=np.int64)
        for _ in range(16):
            d0 = np.linalg.norm(stacked - centers[0], axis=1)
            d1 = np.linalg.norm(stacked - centers[1], axis=1)
            labels = (d1 < d0).astype(np.int64)
            if np.all(labels == 0) or np.all(labels == 1):
                return None, None
            centers = np.stack([stacked[labels == 0].mean(axis=0), stacked[labels == 1].mean(axis=0)], axis=0)
        return labels, centers


# ---------------------------------------------------------------------------
# 7. Systematic mislabel (class-centroid confusion)
# ---------------------------------------------------------------------------


class MislabelDetector(BaseDetector):
    """Flag class-conditional campaigns: many samples sit in another class's E_ref cloud."""

    def __init__(self, margin: float = 1.15, min_campaign: int = 3) -> None:
        super().__init__(
            detector_id="mislabel_detector",
            required_tier="T0",
            supported_tasks=["CLASSIFICATION", "DETECTION", "SEGMENTATION"],
            requires_reference=False,
        )
        self.margin = margin
        self.min_campaign = min_campaign

    def _run(self, dataset: List[Any], capability_matrix: Any, **kwargs: Any) -> List[SampleAnomaly]:
        embeddings: Mapping[str, Any] = kwargs.get("embeddings") or {}
        rows: List[Tuple[Any, str, np.ndarray]] = []
        by_class: Dict[str, List[np.ndarray]] = defaultdict(list)
        for record in dataset:
            emb = _ref_embedding(embeddings, _asset_id(record))
            if emb is None:
                continue
            label = _primary_label(record)
            rows.append((record, label, emb))
            by_class[label].append(emb)
        if len(by_class) < 2:
            return []

        centroids = {
            name: np.stack(vecs, axis=0).mean(axis=0)
            for name, vecs in by_class.items()
            if len(vecs) >= 2
        }
        if len(centroids) < 2:
            return []

        confused: Dict[Tuple[str, str, str], List[Tuple[Any, float, float]]] = defaultdict(list)
        for record, assigned, emb in rows:
            own = centroids.get(assigned)
            if own is None:
                continue
            dist_own = float(np.linalg.norm(emb - own))
            nearest_name = None
            nearest_dist = None
            for name, centroid in centroids.items():
                if name == assigned:
                    continue
                dist = float(np.linalg.norm(emb - centroid))
                if nearest_dist is None or dist < nearest_dist:
                    nearest_name, nearest_dist = name, dist
            if nearest_name is None or nearest_dist is None:
                continue
            if nearest_dist * self.margin >= dist_own:
                continue
            key = (_contributor_id(record), assigned, nearest_name)
            confused[key].append((record, dist_own, nearest_dist))

        findings: List[SampleAnomaly] = []
        for (contributor_id, assigned, nearest), group in confused.items():
            if len(group) < self.min_campaign:
                continue
            for record, dist_own, dist_other in group:
                findings.append(
                    SampleAnomaly(
                        asset_id=_asset_id(record),
                        contributor_id=contributor_id,
                        finding_type=FindingType.SYSTEMATIC_MISLABEL,
                        score=float(min(1.0, 0.4 + 0.1 * len(group) + 0.2 * (dist_own / (dist_other + EPS)))),
                        details={
                            "assigned_label": assigned,
                            "nearest_class": nearest,
                            "dist_assigned": dist_own,
                            "dist_nearest": dist_other,
                            "campaign_size": len(group),
                        },
                    )
                )
        return findings


__all__ = [
    "ActivationClusteringDetector",
    "AnnotationAnomalyDetector",
    "BaseDetector",
    "LabelFlipDetector",
    "MislabelDetector",
    "NearDuplicateDetector",
    "OODInsertionDetector",
    "SpectralSignatureDetector",
]
