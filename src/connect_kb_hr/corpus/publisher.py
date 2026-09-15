"""Shared, product-neutral corpus publisher (unified corpus design, ADR-0005 rev).

The CorpusPublisher validates manifests, chunks content, embeds it, and
atomically activates an immutable release in a product-owned PostgreSQL
database. It is product-neutral: LegalX, SignalX, and HR each supply their
own TargetConfig with separate credentials.

The core has no external DB dependency: the database is reached through the
CorpusStore protocol, mirroring the EntitlementStore pattern in
``gating/adapters.py``. A PostgreSQL adapter implementing the protocol is
supplied by the deployment; tests use an in-memory store.

The publisher handles one target per call. Its store implementation owns the
atomic active-release-pointer update used by activation and rollback.

Invariants enforced here:

- compiled_assignments is non-empty before any mutation;
- every manifest must have at least one matching assignment;
- publication is idempotent for an unchanged manifest + policy;
- the supplied source body must match its manifest content hash before release
  construction;
- the emergency path can suspend/withdraw but never introduce new content;
- events contain identifiers and controlled reason codes, never source bodies,
  DSNs, credentials, or request/response payloads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol, Sequence

from .audit import PublicationEvent, PublicationEventLog
from .chunker import Chunk, DeterministicChunker
from .manifest import CorpusManifest, validate_manifests
from .release import Release, ReleaseBuilder
from .target import TargetConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy_hash(compiled_assignments: list[dict]) -> str:
    """Stable hash of the compiled assignment list."""
    canonical = json.dumps(
        sorted(compiled_assignments, key=lambda a: (
            a.get("source_version_id", ""),
            a.get("audience_code", ""),
            a.get("process_code", ""),
            a.get("content_type_code", ""),
        )),
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_hex(canonical.encode("utf-8"))


# ---------------------------------------------------------------------------
# Embedding protocol — consumer-supplied
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    """Consumer-supplied embedding function pinned by model identity."""

    model_id: str
    model_digest: str

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...


# ---------------------------------------------------------------------------
# CorpusStore protocol — the DB boundary
# ---------------------------------------------------------------------------


class CorpusStore(Protocol):
    """Product-owned corpus database interface the publisher writes through.

    Implementations must be transactional: ``activate_release`` and
    ``rollback_release`` are single atomic operations on the target's
    active-release pointer; readers never observe a partial build.
    """

    def target_schema_version(self) -> str:
        """Return the target's schema version string."""
        ...

    def active_release(self) -> Release | None:
        """Return the currently active release for this target, if any."""
        ...

    def validated_releases(self) -> list[Release]:
        """Return all validated releases, newest first (for rollback)."""
        ...

    def write_release(
        self,
        release: Release,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, list[float]],
        manifests: Sequence[CorpusManifest] | None = None,
        assignments: list[dict] | None = None,
    ) -> None:
        """Persist a built release, chunks/embeddings, and assignments (unactivated).

        ``manifests`` is optional but required for source/source_version FK parents
        to be written. Without it, chunk inserts will fail FK constraints if the
        source_version rows do not already exist.

        ``assignments`` is the compiled assignment list from the kb-hr compiler.
        """
        ...

    def activate_release(self, release_id: str, activated_at: str) -> None:
        """Atomically point this target's active release at ``release_id``."""
        ...

    def rollback_release(self, release_id: str, activated_at: str) -> None:
        """Atomically re-point the active release at a prior validated release."""
        ...

    def set_source_state(self, source_version_id: str, state: str, activated_at: str) -> None:
        """Record a global-state transition (suspend/withdraw) for a version."""
        ...


# ---------------------------------------------------------------------------
# Publication results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationResult:
    """Outcome of one publish attempt against one target."""

    target: TargetConfig
    status: str  # "activated" | "noop" | "failed"
    release_id: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("activated", "noop")


@dataclass(frozen=True)
class EmergencyResult:
    """Outcome of one emergency suspend/withdraw action."""

    target: TargetConfig
    status: str  # "suspended" | "withdrawn" | "failed"
    source_version_id: str
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("suspended", "withdrawn")


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class CorpusPublisher:
    """Builds and activates one target's immutable corpus release."""

    SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

    def __init__(
        self,
        *,
        event_log: PublicationEventLog,
        chunker: DeterministicChunker | None = None,
        embedder: Embedder | None = None,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        self._events = event_log
        self._chunker = chunker or DeterministicChunker()
        self._embedder = embedder
        self._clock = clock

    # -- normal publication path -------------------------------------------

    def publish(
        self,
        *,
        publication_run_id: str,
        target: TargetConfig,
        store: CorpusStore,
        manifests: Sequence[CorpusManifest],
        source_texts: Mapping[str, str],
        source_commit: str,
        compiled_assignments: list[dict],
    ) -> PublicationResult:
        """Publish a manifest batch to one target.

        ``source_texts`` maps source_version_id to normalized source text for
        chunking and embedding. It is consumed in-process only and never
        logged, persisted, or placed in events.

        ``compiled_assignments`` is the list of assignment dicts from the
        kb-hr compiler. It must be non-empty, and every manifest must have at
        least one assignment, before any DB mutation occurs.

        Idempotent: if the target's active release already matches the
        manifest batch hash AND the policy hash, the publish is a no-op.
        """
        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="build_started", timestamp=self._clock(),
            )
        )

        # 1. Gate on target schema compatibility before any mutation.
        try:
            schema_version = store.target_schema_version()
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"schema_check_unavailable:{type(exc).__name__}"
            )
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            return self._fail(
                publication_run_id, target,
                f"unsupported_schema_version:{schema_version}"
            )

        # 2. Validate manifests.
        if not manifests:
            return self._fail(publication_run_id, target, "empty_manifest_batch")
        errors = validate_manifests(list(manifests))
        if errors:
            return self._fail(
                publication_run_id, target,
                f"manifest_validation:{len(errors)}_error(s)"
            )

        # 2a. Validate compiled_assignments gate before any mutation.
        if not compiled_assignments:
            return self._fail(
                publication_run_id, target, "compiled_assignments:empty"
            )
        assigned_svids = {a["source_version_id"] for a in compiled_assignments}
        for m in manifests:
            if m.source_version_id not in assigned_svids:
                return self._fail(
                    publication_run_id, target,
                    f"compiled_assignments:missing_for:{m.source_version_id}"
                )

        # 2b. Require a pinned embedder whose identity matches the batch.
        batch_model_id = manifests[0].embedding_model_id
        batch_model_digest = manifests[0].embedding_model_digest
        if self._embedder is None:
            return self._fail(
                publication_run_id, target, "embedding_configuration:missing_embedder"
            )
        if getattr(self._embedder, "model_id", None) != batch_model_id:
            return self._fail(
                publication_run_id, target, "embedding_configuration:model_id_mismatch"
            )
        if getattr(self._embedder, "model_digest", None) != batch_model_digest:
            return self._fail(
                publication_run_id, target, "embedding_configuration:model_digest_mismatch"
            )

        # 2c. Reject a chunker whose version doesn't match the batch manifest.
        batch_chunker_version = manifests[0].chunker_version
        if self._chunker.version != batch_chunker_version:
            return self._fail(
                publication_run_id,
                target,
                f"chunker_version_mismatch:chunker={self._chunker.version},"
                f"batch={batch_chunker_version}",
            )

        for m in manifests:
            text = source_texts.get(m.source_version_id)
            if text is None:
                return self._fail(publication_run_id, target, "missing_source_text")
            if _sha256_hex(text.encode("utf-8")) != m.content_hash:
                return self._fail(
                    publication_run_id, target, "source_content_hash_mismatch"
                )

        # 3. Compute policy hash and check idempotency.
        ph = _policy_hash(compiled_assignments)
        batch_hash = self._batch_hash(manifests)
        try:
            active = store.active_release()
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"active_release_check_unavailable:{type(exc).__name__}"
            )
        if (
            active is not None
            and active.manifest_hash == batch_hash
            and active.policy_hash == ph
        ):
            self._events.append(
                PublicationEvent(
                    event_id="", publication_run_id=publication_run_id,
                    product=target.product, audience="",
                    kind="idempotent_noop", release_id=active.release_id,
                    manifest_hash=batch_hash, timestamp=self._clock(),
                )
            )
            return PublicationResult(
                target=target, status="noop",
                release_id=active.release_id, detail="unchanged"
            )

        # 4. Chunk and embed.
        try:
            chunks = self._chunk_all(manifests, source_texts)
            embeddings = self._embed_all(chunks)
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"chunking_or_embedding:{type(exc).__name__}"
            )

        # 4a. Validate embedding vectors before write.
        dim_error = self._validate_embeddings(embeddings)
        if dim_error:
            return self._fail(publication_run_id, target, dim_error)

        # 5. Validate: chunk integrity + deterministic release construction.
        result = self._validate_release(
            publication_run_id=publication_run_id,
            target=target,
            manifests=manifests,
            chunks=chunks,
            source_commit=source_commit,
            policy_hash=ph,
            assignment_count=len(compiled_assignments),
        )
        if isinstance(result, PublicationResult):
            # Validation failed — _validate_release already emitted build_failed.
            return result
        release = result
        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="validation_passed", release_id=release.release_id,
                manifest_hash=release.manifest_hash, timestamp=self._clock(),
            )
        )

        # 8. Persist and atomically activate.
        try:
            store.write_release(
                release, chunks, embeddings,
                manifests=list(manifests),
                assignments=compiled_assignments,
            )
            activated = ReleaseBuilder().with_activation(
                release, activated_at=self._clock()
            )
            store.activate_release(activated.release_id, activated.activated_at or "")
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"activation:{type(exc).__name__}",
                release_id=release.release_id,
            )

        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="release_activated", release_id=activated.release_id,
                manifest_hash=activated.manifest_hash, timestamp=self._clock(),
            )
        )
        return PublicationResult(
            target=target, status="activated", release_id=activated.release_id
        )

    # -- emergency path -----------------------------------------------------

    def emergency(
        self,
        *,
        publication_run_id: str,
        target: TargetConfig,
        store: CorpusStore,
        source_version_id: str,
        action: str,
    ) -> EmergencyResult:
        """Suspend or withdraw one source version immediately.

        The emergency path may only remove content from the active corpus:
        ``action`` must be ``"suspend"`` or ``"withdraw"``. It never builds,
        writes, or activates a release, so it cannot introduce new content.
        """
        if action not in ("suspend", "withdraw"):
            return EmergencyResult(
                target=target, status="failed", source_version_id=source_version_id,
                detail=f"emergency_action_not_permitted:{action}",
            )
        state = "suspended" if action == "suspend" else "withdrawn"
        try:
            store.set_source_state(source_version_id, state, self._clock())
        except Exception as exc:
            return EmergencyResult(
                target=target, status="failed", source_version_id=source_version_id,
                detail=f"emergency:{type(exc).__name__}",
            )
        kind = "source_suspended" if action == "suspend" else "source_withdrawn"
        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind=kind, detail=source_version_id, timestamp=self._clock(),
            )
        )
        return EmergencyResult(
            target=target, status=state, source_version_id=source_version_id
        )

    # -- rollback path -------------------------------------------------------

    def rollback(
        self,
        *,
        publication_run_id: str,
        target: TargetConfig,
        store: CorpusStore,
    ) -> PublicationResult:
        """Roll back this target to the immediately previous validated release.

        Reads the store's validated release list and re-points the active
        pointer at the first validated release that is not currently active.
        Never rebuilds, re-chunks, embeds, or mutates content.
        """
        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="build_started", timestamp=self._clock(),
            )
        )

        try:
            active = store.active_release()
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"rollback_active_read:{type(exc).__name__}"
            )

        try:
            validated = store.validated_releases()
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"rollback_validated_read:{type(exc).__name__}"
            )

        active_id = active.release_id if active is not None else None
        predecessor = next(
            (r for r in validated if r.release_id != active_id),
            None,
        )
        if predecessor is None:
            return self._fail(publication_run_id, target, "no_prior_validated_release")

        try:
            store.rollback_release(predecessor.release_id, self._clock())
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"rollback_write:{type(exc).__name__}",
                release_id=predecessor.release_id,
            )

        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="release_rolled_back",
                release_id=predecessor.release_id,
                manifest_hash=predecessor.manifest_hash,
                timestamp=self._clock(),
            )
        )
        return PublicationResult(
            target=target, status="activated", release_id=predecessor.release_id
        )

    # -- helpers -------------------------------------------------------------

    def _validate_release(
        self,
        *,
        publication_run_id: str,
        target: TargetConfig,
        manifests: Sequence[CorpusManifest],
        chunks: list[Chunk],
        source_commit: str,
        policy_hash: str,
        assignment_count: int,
    ) -> Release | PublicationResult:
        """Run all explicit validation gates and return a validated Release or a failure result."""
        chunk_errors = DeterministicChunker.verify_chunks(chunks)
        if chunk_errors:
            return self._fail(
                publication_run_id, target,
                f"chunk_verification:{len(chunk_errors)}_error(s)"
            )

        builder = ReleaseBuilder()
        try:
            release = builder.build(
                publication_run_id=publication_run_id,
                product=target.product,
                manifests=list(manifests),
                chunks=chunks,
                source_commit=source_commit,
                policy_hash=policy_hash,
                assignment_count=assignment_count,
            )
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"release_construction:{type(exc).__name__}"
            )

        try:
            release = builder.with_validation(release, passed=True)
        except Exception as exc:
            return self._fail(
                publication_run_id, target,
                f"release_validation:{type(exc).__name__}"
            )

        return release

    def _chunk_all(
        self, manifests: Sequence[CorpusManifest], source_texts: Mapping[str, str]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for m in manifests:
            text = source_texts[m.source_version_id]
            chunks.extend(
                self._chunker.chunk(
                    m.source_version_id, m.source_kind, text,
                    locator_prefix=m.canonical_locator,
                )
            )
        return chunks

    def _embed_all(self, chunks: Sequence[Chunk]) -> dict[str, list[float]]:
        if self._embedder is None:
            return {}
        return {c.chunk_id: self._embedder.embed(c.content) for c in chunks}

    @staticmethod
    def _validate_embeddings(embeddings: Mapping[str, object]) -> str | None:
        """Check that all embedding vectors are finite numeric lists of one dimension."""
        if not embeddings:
            return None
        dims: set[int] = set()
        for vec in embeddings.values():
            if not isinstance(vec, list):
                return "embedding_validation:invalid_vector"
            if len(vec) == 0:
                return "embedding_validation:empty_vector"
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in vec
            ):
                return "embedding_validation:invalid_vector"
            if not all(
                value == value and value not in (float("inf"), float("-inf"))
                for value in vec
            ):
                return "embedding_validation:non_finite_values"
            dims.add(len(vec))
        if len(dims) > 1:
            return "embedding_validation:inconsistent_dimension"
        return None

    def _fail(
        self,
        publication_run_id: str,
        target: TargetConfig,
        detail: str,
        release_id: str | None = None,
    ) -> PublicationResult:
        self._events.append(
            PublicationEvent(
                event_id="", publication_run_id=publication_run_id,
                product=target.product, audience="",
                kind="build_failed", release_id=release_id,
                detail=detail, timestamp=self._clock(),
            )
        )
        return PublicationResult(
            target=target, status="failed",
            release_id=release_id, detail=detail
        )

    @staticmethod
    def _batch_hash(manifests: Sequence[CorpusManifest]) -> str:
        joined = "|".join(
            m.manifest_hash
            for m in sorted(manifests, key=lambda m: m.source_version_id)
        )
        return _sha256_hex(joined.encode("utf-8"))
