"""M1 data-integrity schemas (CV-ASSURE v2, Clause 2.2.1).

These types are the contract between sample-level detectors, contributor
rollups, Sybil clustering, and the M5 risk-fusion layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FindingType(str, Enum):
    """Sample-level data-integrity finding categories (Clause 2.2.1)."""

    TRIGGER_INJECTION = "TRIGGER_INJECTION"
    """Backdoor / trigger pattern inserted into inputs or labels."""

    LABEL_FLIP = "LABEL_FLIP"
    """Individual samples whose labels disagree with local neighbourhood evidence."""

    SYSTEMATIC_MISLABEL = "SYSTEMATIC_MISLABEL"
    """Class-conditional or contributor-conditional mislabel campaign."""

    NEAR_DUPLICATE_FLOOD = "NEAR_DUPLICATE_FLOOD"
    """Near-duplicate images used to inflate a class or contributor."""

    OOD_INSERTION = "OOD_INSERTION"
    """Out-of-distribution samples inserted into the training corpus."""

    ANNOTATION_ANOMALY = "ANNOTATION_ANOMALY"
    """Malformed or statistically impossible boxes / polygons / attributes."""

    DATA_MODEL_COLLUSION = "DATA_MODEL_COLLUSION"
    """Novel E_ref vs E_sub divergence: data and subject model appear collusive."""


Disposition = Literal["quarantine", "review", "accept"]


class SampleAnomaly(BaseModel):
    """Per-asset detector finding produced by an M1 integrity check."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1, description="AssetRecord.asset_id of the flagged sample.")
    contributor_id: str = Field(..., min_length=1, description="Uploader / dataset contributor.")
    finding_type: FindingType = Field(..., description="Clause 2.2.1 finding category.")
    score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score in [0, 1].")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Detector-specific evidence, e.g. k-NN label mismatch, "
            "Mahalanobis distance, pHash distance, E_ref/E_sub cosine."
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict with enum values as strings."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize this finding to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleAnomaly":
        """Build a ``SampleAnomaly`` from a plain dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str) -> "SampleAnomaly":
        """Build a ``SampleAnomaly`` from a JSON string."""
        return cls.model_validate_json(payload)


class ContributorRollup(BaseModel):
    """Contributor-level binomial rollup of sample findings (Clause 2.2.1)."""

    model_config = ConfigDict(extra="forbid")

    contributor_id: str = Field(..., min_length=1)
    total_samples: int = Field(..., ge=0)
    flagged_samples: int = Field(..., ge=0)
    flag_rate: float = Field(..., ge=0.0, le=1.0, description="flagged_samples / total_samples.")
    pipeline_baseline_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expected flag rate under the clean / pipeline baseline.",
    )
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Exact binomial test p-value vs pipeline_baseline_rate.",
    )
    estimated_poison_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Point estimate of contributor poison fraction.",
    )
    concentrated_classes: List[str] = Field(
        default_factory=list,
        description="Classes that absorb a disproportionate share of flags.",
    )
    disposition: Disposition = Field(
        ...,
        description="quarantine | review | accept",
    )
    reason: str = Field(
        default="",
        description="Clause 2.2.1 narrative: flag rate vs baseline, p-value, class concentration, poison CI.",
    )
    sybil_cluster_id: Optional[str] = Field(
        default=None,
        description="Shared cluster id when this contributor is grouped with Sybils.",
    )

    @model_validator(mode="after")
    def _flagged_not_above_total(self) -> "ContributorRollup":
        if self.flagged_samples > self.total_samples:
            raise ValueError("flagged_samples cannot exceed total_samples")
        if self.total_samples == 0 and self.flag_rate != 0.0:
            raise ValueError("flag_rate must be 0 when total_samples is 0")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize this rollup to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContributorRollup":
        """Build a ``ContributorRollup`` from a plain dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str) -> "ContributorRollup":
        """Build a ``ContributorRollup`` from a JSON string."""
        return cls.model_validate_json(payload)


class CapabilityMatrix(BaseModel):
    """Capability probe snapshot attached to an M1 result.

    Accepts the M0 ``CapabilityMatrix`` (``to_dict()`` / attributes) or a
    plain mapping so M1 does not import ONNX at schema load time.
    """

    model_config = ConfigDict(extra="allow")

    access_tier: str = Field(..., description="T0 / T1 / T2 (or v2 names such as T0_LABELS).")
    task_type: str = Field(..., description="CLASSIFICATION | DETECTION | SEGMENTATION.")
    reference_mode: str = Field(..., description="ATTESTED | BOOTSTRAPPED | UNREFERENCED.")
    model_digest: Optional[str] = None
    reference_profile_digest: Optional[str] = None
    probe_timestamp: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("access_tier", "task_type", "reference_mode", mode="before")
    @classmethod
    def _enum_to_str(cls, value: Any) -> str:
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    @classmethod
    def from_m0(cls, matrix: Any) -> "CapabilityMatrix":
        """Coerce an M0 capability matrix (object or dict) into this snapshot."""
        if isinstance(matrix, cls):
            return matrix
        if isinstance(matrix, dict):
            return cls.model_validate(matrix)
        if hasattr(matrix, "to_dict") and callable(matrix.to_dict):
            return cls.model_validate(matrix.to_dict())
        payload = {
            "access_tier": getattr(matrix, "access_tier", "T0"),
            "task_type": getattr(matrix, "task_type", "DETECTION"),
            "reference_mode": getattr(matrix, "reference_mode", "UNREFERENCED"),
            "model_digest": getattr(matrix, "model_digest", None),
            "reference_profile_digest": getattr(matrix, "reference_profile_digest", None),
            "probe_timestamp": getattr(matrix, "probe_timestamp", None),
            "source": getattr(matrix, "source", None),
            "metadata": getattr(matrix, "metadata", None) or {},
        }
        return cls.model_validate(payload)


class M1Result(BaseModel):
    """Aggregate output of Module M1 for fusion and reporting."""

    model_config = ConfigDict(extra="forbid")

    capability_matrix: CapabilityMatrix = Field(
        ...,
        description="Probe result that gated which M1 detectors were allowed to run.",
    )
    sample_anomalies: List[SampleAnomaly] = Field(default_factory=list)
    contributor_rollups: List[ContributorRollup] = Field(default_factory=list)
    sybil_suspect_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of contributor_ids believed to be Sybil identities.",
    )

    @field_validator("capability_matrix", mode="before")
    @classmethod
    def _coerce_matrix(cls, value: Any) -> Any:
        if isinstance(value, CapabilityMatrix):
            return value
        return CapabilityMatrix.from_m0(value)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize this result to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M1Result":
        """Build an ``M1Result`` from a plain dict."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: str) -> "M1Result":
        """Build an ``M1Result`` from a JSON string."""
        return cls.model_validate_json(payload)


MatrixLike = Union[CapabilityMatrix, Dict[str, Any], Any]
