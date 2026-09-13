"""Corpus manifest contract (issue #20, ADR-0005 sections 3-4).

A CorpusManifest is the versioned publication record for one immutable source
version. It carries the full provenance, rights, and global-state metadata the
publisher needs to validate publication eligibility. The manifest is the
shared contract between canonical origins (LegalX repository, RHH repository)
and the publisher; it never contains source body text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

GlobalState = Literal["valid", "suspended", "withdrawn", "superseded"]

MANIFEST_SCHEMA_VERSION = "1.0"

_VALID_STATES: tuple[str, ...] = ("valid", "suspended", "withdrawn", "superseded")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EffectiveInterval:
    """Half-open effective interval [start, end) for a source version.

    ``end=None`` means the version is currently effective with no scheduled
    expiry. Dates are ISO-8601 strings so the manifest stays serializable
    without datetime dependencies.
    """

    start: str
    end: str | None = None


@dataclass(frozen=True)
class Provenance:
    """Where and how a source version was produced.

    ``origin_commit`` is the canonical repository revision (e.g. a git SHA)
    the version was compiled from. ``reviewed_by`` and ``reviewed_at`` record
    the product/audience approval that admitted this version to publication.
    """

    origin_commit: str
    reviewed_by: str | None = None
    reviewed_at: str | None = None


@dataclass(frozen=True)
class CorpusManifest:
    """Versioned publication manifest for one immutable source version.

    Every field is required publication metadata per ADR-0005 section 3.
    The manifest is content-free: it records hashes of the substantive
    content and metadata, never the content itself.
    """

    source_id: str
    source_version_id: str
    source_kind: str
    canonical_origin: str
    content_hash: str
    metadata_hash: str
    jurisdiction: str
    effective_interval: EffectiveInterval
    provenance: Provenance
    canonical_locator: str
    rights_basis: str
    rights_holder: str
    permitted_use: str
    attribution_required: bool
    global_state: GlobalState
    chunker_version: str
    embedding_model_id: str
    embedding_model_digest: str
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.global_state not in _VALID_STATES:
            raise ValueError(f"unknown global_state: {self.global_state!r}")
        if not self.source_id or not self.source_version_id:
            raise ValueError("source_id and source_version_id are required")
        if not self.content_hash:
            raise ValueError("content_hash is required")
        if not self.rights_basis or not self.rights_holder:
            raise ValueError("rights_basis and rights_holder are required publication gates")
        if not self.canonical_origin or not self.canonical_locator:
            raise ValueError("canonical_origin and canonical_locator are required")
        for field_name in ("chunker_version", "embedding_model_id", "embedding_model_digest"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} is required")
        if self.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest_schema_version: {self.manifest_schema_version!r}")

    @property
    def is_publishable(self) -> bool:
        """Only globally valid sources may enter a new active release."""
        return self.global_state == "valid"

    @property
    def manifest_hash(self) -> str:
        """Stable identity hash over the full manifest.

        Used for idempotency: re-publishing a manifest whose
        ``manifest_hash`` matches the active release is a no-op.
        """
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))

    def to_canonical_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, no whitespace)."""
        payload: dict[str, object] = {
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "source_kind": self.source_kind,
            "canonical_origin": self.canonical_origin,
            "content_hash": self.content_hash,
            "metadata_hash": self.metadata_hash,
            "jurisdiction": self.jurisdiction,
            "effective_interval": {
                "start": self.effective_interval.start,
                "end": self.effective_interval.end,
            },
            "provenance": {
                "origin_commit": self.provenance.origin_commit,
                "reviewed_by": self.provenance.reviewed_by,
                "reviewed_at": self.provenance.reviewed_at,
            },
            "canonical_locator": self.canonical_locator,
            "rights_basis": self.rights_basis,
            "rights_holder": self.rights_holder,
            "permitted_use": self.permitted_use,
            "attribution_required": self.attribution_required,
            "global_state": self.global_state,
            "chunker_version": self.chunker_version,
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_digest": self.embedding_model_digest,
            "manifest_schema_version": self.manifest_schema_version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_manifests(manifests: Sequence[CorpusManifest]) -> list[str]:
    """Validate a batch of manifests for a publication run.

    Returns a list of human-readable error strings (empty when the batch is
    clean). Errors never include source body text.

    All manifests in a batch must share the same ``chunker_version``,
    ``embedding_model_id``, and ``embedding_model_digest``; mixing them
    across sources would produce an incoherent release.
    """
    errors: list[str] = []
    seen: dict[str, str] = {}
    for m in manifests:
        prefix = f"source {m.source_id!r} version {m.source_version_id!r}"
        if not m.is_publishable:
            errors.append(f"{prefix}: global_state {m.global_state!r} is not publishable")
        if m.source_version_id in seen:
            errors.append(f"duplicate source_version_id {m.source_version_id!r}")
        seen[m.source_version_id] = m.source_id

    # Batch-level compatibility: all manifests must agree on chunker/embedding identity.
    for field_name in ("chunker_version", "embedding_model_id", "embedding_model_digest"):
        values = {getattr(m, field_name) for m in manifests}
        if len(values) > 1:
            errors.append(f"mixed {field_name} in batch: {sorted(values)}")

    return errors
