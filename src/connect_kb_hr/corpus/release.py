"""Immutable corpus releases (issue #20, ADR-0005 section 12).

A Release is an immutable, audience-specific publication unit. It is built
outside the active read path from validated manifests, validated, and
activated by transactionally changing one target's active-release pointer.
Releases are never mutated in place; rollback re-points to the prior
validated release.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .chunker import Chunk
from .manifest import CorpusManifest

BuildStatus = Literal["building", "built", "failed"]
ValidationStatus = Literal["pending", "passed", "failed"]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Release:
    """One immutable, product/audience-specific corpus release.

    ``release_id`` is derived from the publication run, target, and manifest
    hashes, so rebuilding the same release yields the same ID.
    """

    release_id: str
    publication_run_id: str
    product: str
    audience: str
    manifest_hash: str
    source_commit: str
    chunker_version: str
    embedding_model_id: str
    build_status: BuildStatus
    validation_status: ValidationStatus
    activated_at: str | None = None
    chunk_count: int = 0

    def __post_init__(self) -> None:
        if self.build_status not in ("building", "built", "failed"):
            raise ValueError(f"unknown build_status: {self.build_status!r}")
        if self.validation_status not in ("pending", "passed", "failed"):
            raise ValueError(f"unknown validation_status: {self.validation_status!r}")

    @property
    def is_activated(self) -> bool:
        return self.activated_at is not None

    @property
    def is_validated(self) -> bool:
        return self.validation_status == "passed"


def _release_id(
    publication_run_id: str,
    product: str,
    audience: str,
    manifest_hash: str,
    chunker_version: str,
    embedding_model_id: str,
) -> str:
    basis = f"{publication_run_id}|{product}|{audience}|{manifest_hash}|{chunker_version}|{embedding_model_id}"
    return _sha256_hex(basis.encode("utf-8"))


class ReleaseBuilder:
    """Builds immutable releases from validated manifests and chunks."""

    def build(
        self,
        *,
        publication_run_id: str,
        product: str,
        audience: str,
        manifests: Sequence[CorpusManifest],
        chunks: Sequence[Chunk],
        source_commit: str,
    ) -> Release:
        """Assemble a release from a validated manifest batch.

        The release is created in ``built``/``pending`` state; the publisher
        promotes validation status before activation.
        """
        if not manifests:
            raise ValueError("cannot build a release from zero manifests")
        manifest_hash = _sha256_hex(
            "|".join(m.manifest_hash for m in sorted(manifests, key=lambda m: m.source_version_id)).encode("utf-8")
        )
        chunker_version = manifests[0].chunker_version
        embedding_model_id = manifests[0].embedding_model_id
        return Release(
            release_id=_release_id(
                publication_run_id,
                product,
                audience,
                manifest_hash,
                chunker_version,
                embedding_model_id,
            ),
            publication_run_id=publication_run_id,
            product=product,
            audience=audience,
            manifest_hash=manifest_hash,
            source_commit=source_commit,
            chunker_version=chunker_version,
            embedding_model_id=embedding_model_id,
            build_status="built",
            validation_status="pending",
            chunk_count=len(chunks),
        )

    def with_validation(self, release: Release, *, passed: bool, reason: str | None = None) -> Release:
        """Return a copy of ``release`` with validation status set."""
        return Release(
            release_id=release.release_id,
            publication_run_id=release.publication_run_id,
            product=release.product,
            audience=release.audience,
            manifest_hash=release.manifest_hash,
            source_commit=release.source_commit,
            chunker_version=release.chunker_version,
            embedding_model_id=release.embedding_model_id,
            build_status=release.build_status,
            validation_status="passed" if passed else "failed",
            activated_at=release.activated_at,
            chunk_count=release.chunk_count,
        )

    def with_activation(self, release: Release, activated_at: str) -> Release:
        """Return a copy of ``release`` marked activated at ``activated_at``."""
        if not release.is_validated:
            raise ValueError("cannot activate a release that has not passed validation")
        return Release(
            release_id=release.release_id,
            publication_run_id=release.publication_run_id,
            product=release.product,
            audience=release.audience,
            manifest_hash=release.manifest_hash,
            source_commit=release.source_commit,
            chunker_version=release.chunker_version,
            embedding_model_id=release.embedding_model_id,
            build_status=release.build_status,
            validation_status=release.validation_status,
            activated_at=activated_at,
            chunk_count=release.chunk_count,
        )
