"""InMemoryCorpusStore — unit-testable CorpusStore for publish pipeline tests.

Implements the full CorpusStore protocol in memory. Used in tests that
validate publisher logic without a live Postgres instance.

Audience filtering is applied in-memory via the assignment lookup table,
mirroring the SQL join in PostgresCorpusStore.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from connect_kb_hr.corpus.chunker import Chunk
from connect_kb_hr.corpus.publisher import CorpusStore
from connect_kb_hr.corpus.release import Release

SUPPORTED_SCHEMA_VERSION = "1.0"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryCorpusStore:
    """In-memory CorpusStore; mirrors the PostgresCorpusStore contract exactly.

    Assignments are stored in ``_assignments``: a list of dicts with keys
    release_id, source_version_id, audience_code, process_code,
    content_type_code, source_role_code.
    """

    def __init__(self, schema_version: str = SUPPORTED_SCHEMA_VERSION) -> None:
        self._schema_version = schema_version
        self._releases: dict[str, Release] = {}
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._active_release_id: str | None = None
        self._source_states: dict[str, str] = {}
        self._assignments: list[dict] = []
        # publication outbox events (content-free tags only)
        self.outbox_events: list[dict] = []

    # ------------------------------------------------------------------
    # CorpusStore protocol
    # ------------------------------------------------------------------

    def target_schema_version(self) -> str:
        return self._schema_version

    def active_release(self) -> Release | None:
        if self._active_release_id is None:
            return None
        return self._releases.get(self._active_release_id)

    def validated_releases(self) -> list[Release]:
        return sorted(
            [r for r in self._releases.values() if r.validation_status == "passed"],
            key=lambda r: r.release_id,
            reverse=True,
        )

    def write_release(
        self,
        release: Release,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, list[float]],
        manifests=None,  # accepted but not needed in-memory
        assignments: list[dict] | None = None,
    ) -> None:
        # Idempotent: ON CONFLICT DO NOTHING semantics
        if release.release_id not in self._releases:
            self._releases[release.release_id] = release
        for chunk in chunks:
            if chunk.chunk_id not in self._chunks:
                self._chunks[chunk.chunk_id] = chunk
        for chunk_id, vec in embeddings.items():
            if chunk_id not in self._embeddings:
                self._embeddings[chunk_id] = vec
        # Store assignments; idempotent by (release_id, source_version_id, audience_code,
        # process_code, content_type_code) composite
        if assignments:
            existing_keys = {
                (a["release_id"], a["source_version_id"], a["audience_code"],
                 a["process_code"], a["content_type_code"])
                for a in self._assignments
            }
            for a in assignments:
                a_with_release = dict(a)
                a_with_release.setdefault("release_id", release.release_id)
                key = (
                    a_with_release["release_id"],
                    a_with_release["source_version_id"],
                    a_with_release["audience_code"],
                    a_with_release["process_code"],
                    a_with_release["content_type_code"],
                )
                if key not in existing_keys:
                    self._assignments.append(a_with_release)
                    existing_keys.add(key)

    def activate_release(self, release_id: str, activated_at: str) -> None:
        from connect_kb_hr.corpus.release import ReleaseBuilder
        release = self._releases[release_id]
        activated = ReleaseBuilder().with_activation(release, activated_at)
        self._releases[release_id] = activated
        self._active_release_id = release_id
        # Emit content-free outbox event
        self.outbox_events.append({
            "event_type": "release_activated",
            "release_id": release_id,
            "activated_at": activated_at,
        })

    def rollback_release(self, release_id: str, activated_at: str) -> None:
        self.activate_release(release_id, activated_at)

    def set_source_state(
        self, source_version_id: str, state: str, activated_at: str
    ) -> None:
        self._source_states[source_version_id] = state

    def search(
        self,
        vector: list[float],
        process: str,
        content_type: str,
        audience: str,
        limit: int = 5,
    ) -> list[dict]:
        """Cosine similarity search over chunks in the active release.

        Filters by audience_code, process_code, content_type_code in the
        assignment table, and excludes suspended/withdrawn source versions.
        """
        active = self.active_release()
        if active is None:
            return []

        # Build assignment lookup for the active release
        release_id = active.release_id
        assigned_keys: set[tuple[str, str]] = set()
        assignment_meta: dict[tuple[str, str], dict] = {}
        for a in self._assignments:
            if (
                a.get("release_id", release_id) == release_id
                and a.get("audience_code") == audience
                and a.get("process_code") == process
                and a.get("content_type_code") == content_type
            ):
                sv_id = a["source_version_id"]
                # key: (source_version_id,) — a chunk qualifies if its sv is assigned
                assigned_keys.add(sv_id)
                assignment_meta[sv_id] = a

        results = []
        for chunk in self._chunks.values():
            # Only chunks whose release matches the active release
            if chunk.chunk_id not in self._embeddings:
                continue
            # Skip suspended/withdrawn source versions
            state = self._source_states.get(chunk.source_version_id, "valid")
            if state != "valid":
                continue
            # Filter by assignment predicates
            if chunk.source_version_id not in assigned_keys:
                continue
            sim = _cosine(vector, self._embeddings[chunk.chunk_id])
            a_meta = assignment_meta.get(chunk.source_version_id, {})
            results.append({
                "chunk_id": chunk.chunk_id,
                "source_version_id": chunk.source_version_id,
                "source_locator": chunk.source_locator,
                "content_hash": chunk.content_hash,
                "similarity": sim,
                "audience": audience,
                "process": process,
                "content_type": content_type,
                "source_role": a_meta.get("source_role_code"),
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:limit]
