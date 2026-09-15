"""Release management for the unified corpus design.

A Release represents an immutable publication of the corpus. It is built
from a manifest, chunks, embeddings, and a policy hash. Audience is
expressed as compiled assignments in kb.release_assignments, not as a
field on the Release itself (ADR-0005).

The ReleaseBuilder pattern ensures that release creation is atomic and
verifiable via policy_hash.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from connect_kb_hr.corpus.chunker import Chunk
from connect_kb_hr.corpus.manifest import CorpusManifest


@dataclass(frozen=True)
class Release:
    """Immutable publication of the corpus.

    Fields:
        release_id: Unique identifier (UUID v4).
        publication_run_id: Identifies the publication run that produced this release.
        product: The product this release belongs to.
        manifest_hash: SHA256 hash of the manifest JSON.
        policy_hash: SHA256 hash of the compiled assignments (see ReleaseBuilder.build).
        source_commit: Git commit that produced the source data.
        chunker_version: Version of the chunker used.
        embedding_model_id: Identifier of the embedding model.
        build_status: 'building', 'built', or 'failed'.
        validation_status: 'pending', 'passed', or 'failed'.
        activated_at: When this release was activated (None if not active).
        chunk_count: Number of chunks in this release.
        assignment_count: Number of assignment rows in kb.release_assignments.

    Audience is NOT a field here. Audience is a predicate in kb.release_assignments.
    """

    release_id: str
    publication_run_id: str
    product: str
    manifest_hash: str
    policy_hash: str
    source_commit: str
    chunker_version: str
    embedding_model_id: str
    build_status: str
    validation_status: str
    activated_at: str | None
    chunk_count: int
    assignment_count: int

    def __post_init__(self) -> None:
        if self.build_status not in ("building", "built", "failed"):
            raise ValueError(f"build_status must be 'building'|'built'|'failed', got {self.build_status}")
        if self.validation_status not in ("pending", "passed", "failed"):
            raise ValueError(f"validation_status must be 'pending'|'passed'|'failed', got {self.validation_status}")

    @property
    def is_active(self) -> bool:
        """True if this release is the current active release."""
        return self.activated_at is not None

    @property
    def is_validated(self) -> bool:
        """True if validation passed."""
        return self.validation_status == "passed"

    def __repr__(self) -> str:
        return (
            f"Release(id={self.release_id}, product={self.product!r}, "
            f"build_status={self.build_status!r}, "
            f"validation_status={self.validation_status!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "release_id": self.release_id,
            "publication_run_id": self.publication_run_id,
            "product": self.product,
            "manifest_hash": self.manifest_hash,
            "policy_hash": self.policy_hash,
            "source_commit": self.source_commit,
            "chunker_version": self.chunker_version,
            "embedding_model_id": self.embedding_model_id,
            "build_status": self.build_status,
            "validation_status": self.validation_status,
            "activated_at": self.activated_at,
            "chunk_count": self.chunk_count,
            "assignment_count": self.assignment_count,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Release":
        """Reconstruct from a dict (for migrations)."""
        return Release(
            release_id=data["release_id"],
            publication_run_id=data["publication_run_id"],
            product=data["product"],
            manifest_hash=data["manifest_hash"],
            policy_hash=data["policy_hash"],
            source_commit=data["source_commit"],
            chunker_version=data["chunker_version"],
            embedding_model_id=data["embedding_model_id"],
            build_status=data["build_status"],
            validation_status=data["validation_status"],
            activated_at=data.get("activated_at"),
            chunk_count=data.get("chunk_count", 0),
            assignment_count=data.get("assignment_count", 0),
        )


def _sha256_hex(data: bytes) -> str:
    """Return lowercase hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


class ReleaseBuilder:
    """Stateless factory for building immutable releases.

    Audience is NOT a field on Release — it lives in release_assignments.
    policy_hash is derived from the compiled_assignments list.
    """

    def build(
        self,
        *,
        publication_run_id: str,
        product: str,
        manifests: Sequence[CorpusManifest],
        chunks: Sequence[Chunk],
        source_commit: str,
        policy_hash: str,
        assignment_count: int,
    ) -> Release:
        """Assemble a Release from validated manifests, chunks, and policy."""
        if not manifests:
            raise ValueError("cannot build a release from zero manifests")

        manifest_hash = _sha256_hex(
            "|".join(
                m.manifest_hash
                for m in sorted(manifests, key=lambda m: m.source_version_id)
            ).encode("utf-8")
        )
        chunker_version = manifests[0].chunker_version
        embedding_model_id = manifests[0].embedding_model_id

        release_id = _sha256_hex(
            f"{publication_run_id}|{product}|{manifest_hash}|{policy_hash}|{chunker_version}|{embedding_model_id}".encode(
                "utf-8"
            )
        )

        return Release(
            release_id=release_id,
            publication_run_id=publication_run_id,
            product=product,
            manifest_hash=manifest_hash,
            policy_hash=policy_hash,
            source_commit=source_commit,
            chunker_version=chunker_version,
            embedding_model_id=embedding_model_id,
            build_status="built",
            validation_status="pending",
            activated_at=None,
            chunk_count=len(chunks),
            assignment_count=assignment_count,
        )

    def with_validation(self, release: Release, *, passed: bool, reason: str | None = None) -> Release:
        """Return a copy of ``release`` with validation status set."""
        return Release(
            release_id=release.release_id,
            publication_run_id=release.publication_run_id,
            product=release.product,
            manifest_hash=release.manifest_hash,
            policy_hash=release.policy_hash,
            source_commit=release.source_commit,
            chunker_version=release.chunker_version,
            embedding_model_id=release.embedding_model_id,
            build_status=release.build_status,
            validation_status="passed" if passed else "failed",
            activated_at=release.activated_at,
            chunk_count=release.chunk_count,
            assignment_count=release.assignment_count,
        )

    def with_activation(self, release: Release, activated_at: str) -> Release:
        """Return a copy of ``release`` marked activated."""
        if not release.is_validated:
            raise ValueError("cannot activate a release that has not passed validation")
        return Release(
            release_id=release.release_id,
            publication_run_id=release.publication_run_id,
            product=release.product,
            manifest_hash=release.manifest_hash,
            policy_hash=release.policy_hash,
            source_commit=release.source_commit,
            chunker_version=release.chunker_version,
            embedding_model_id=release.embedding_model_id,
            build_status=release.build_status,
            validation_status=release.validation_status,
            activated_at=activated_at,
            chunk_count=release.chunk_count,
            assignment_count=release.assignment_count,
        )
