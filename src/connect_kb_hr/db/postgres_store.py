"""PostgreSQL CorpusStore adapter — unified kb schema.

Implements the CorpusStore protocol defined in publisher.py against the
unified ``kb`` schema. Audience is an assignment predicate in
``kb.release_assignments``, not a schema or DSN selector.

Credentials come from environment variables only (CORPUS_{PRODUCT}_DSN).
DSNs are never logged or exposed in repr output.

The store is instantiated once per publication run; it does not hold a
long-lived connection pool (the publisher is a batch process, not a server).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import psycopg2
import psycopg2.extras

from connect_kb_hr.corpus.chunker import Chunk
from connect_kb_hr.corpus.manifest import CorpusManifest
from connect_kb_hr.corpus.release import Release

# Schema version this adapter was written against.
# The publisher gates on this before any mutation.
SUPPORTED_SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class PostgresCorpusStore:
    """PostgreSQL implementation of CorpusStore for the unified kb schema.

    ``dsn`` is taken from environment variables; ``schema`` defaults to
    ``'kb'``. Never log, print, or repr the dsn.
    """

    def __init__(self, dsn: str, schema: str = "kb") -> None:
        if not dsn:
            raise ValueError("dsn is required (never logged)")
        self._dsn = dsn
        self._schema = schema

    def __repr__(self) -> str:
        return f"PostgresCorpusStore(schema={self._schema!r}, dsn=<redacted>)"

    @classmethod
    def from_env(cls, product: str = "hr") -> PostgresCorpusStore:
        """Build a store from CORPUS_{PRODUCT}_DSN env var."""
        key = f"CORPUS_{product.upper()}_DSN"
        dsn = os.environ.get(key, "")
        if not dsn:
            raise ValueError(f"{key} is required (no default)")
        return cls(dsn=dsn, schema="kb")

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        return conn

    # ------------------------------------------------------------------
    # CorpusStore protocol
    # ------------------------------------------------------------------

    def target_schema_version(self) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT version FROM {self._schema}.schema_version "
                    "ORDER BY applied_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(
                        f"No schema_version row found in {self._schema}"
                    )
                return row[0]

    def active_release(self) -> Release | None:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT r.*
                    FROM {self._schema}.releases r
                    JOIN {self._schema}.active_release ar ON ar.release_id = r.release_id
                """)
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_release(row)

    def validated_releases(self) -> list[Release]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT *
                    FROM {self._schema}.releases
                    WHERE validation_status = 'passed'
                    ORDER BY created_at DESC
                """)
                return [self._row_to_release(row) for row in cur.fetchall()]

    def write_release(
        self,
        release: Release,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, list[float]],
        manifests: Sequence[CorpusManifest] | None = None,
        assignments: list[dict] | None = None,
    ) -> None:
        """Write release + source projections + chunks + embeddings + assignments atomically.

        ``manifests`` is optional but required for sources/source_versions to
        be written. Without it, FK constraints on chunks will fail if the
        source_version rows do not already exist.

        ``assignments`` is a list of dicts with keys:
            source_version_id, audience_code, process_code, content_type_code,
            source_role_code, approval_record, reviewer_id, reviewed_at

        Idempotent: ON CONFLICT DO NOTHING throughout.

        Note: chunks in kb schema are global (no release_id column); they are
        shared across releases and keyed by (source_version_id, ordinal).
        The release_assignments table links a release to source_versions
        for audience/process/content_type filtering at query time.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                # 1. Upsert source + source_version rows (FK parents of chunks)
                if manifests:
                    for m in manifests:
                        cur.execute(f"""
                            INSERT INTO {self._schema}.sources (
                                source_id, source_kind, canonical_origin,
                                rights_basis, rights_holder, permitted_use,
                                attribution_required, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                            ON CONFLICT (source_id) DO NOTHING
                        """, (
                            m.source_id, m.source_kind, m.canonical_origin,
                            m.rights_basis, m.rights_holder, m.permitted_use,
                            m.attribution_required,
                        ))
                        cur.execute(f"""
                            INSERT INTO {self._schema}.source_versions (
                                source_version_id, source_id, content_hash, metadata_hash,
                                jurisdiction, effective_start, effective_end,
                                canonical_locator, origin_commit, global_state,
                                chunker_version, embedding_model_id, embedding_model_digest,
                                manifest_schema_version, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                            ON CONFLICT (source_version_id) DO NOTHING
                        """, (
                            m.source_version_id, m.source_id,
                            m.content_hash, m.metadata_hash,
                            m.jurisdiction,
                            m.effective_interval.start,
                            m.effective_interval.end,
                            m.canonical_locator,
                            m.provenance.origin_commit,
                            m.global_state,
                            m.chunker_version,
                            m.embedding_model_id,
                            m.embedding_model_digest,
                            m.manifest_schema_version,
                        ))

                # 2. Insert release record
                cur.execute(f"""
                    INSERT INTO {self._schema}.releases (
                        release_id, publication_run_id, manifest_hash, policy_hash,
                        source_commit, chunker_version, embedding_model_id,
                        build_status, validation_status,
                        chunk_count, assignment_count, activated_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (release_id) DO NOTHING
                """, (
                    release.release_id,
                    release.publication_run_id,
                    release.manifest_hash,
                    release.policy_hash,
                    release.source_commit,
                    release.chunker_version,
                    release.embedding_model_id,
                    release.build_status,
                    release.validation_status,
                    release.chunk_count,
                    release.assignment_count,
                    release.activated_at,
                ))

                # 3. Insert chunks + embeddings (global, not per-release)
                for chunk in chunks:
                    vec = embeddings.get(chunk.chunk_id)
                    vec_sql = (
                        f"[{','.join(str(v) for v in vec)}]" if vec is not None else None
                    )
                    cur.execute(f"""
                        INSERT INTO {self._schema}.chunks (
                            chunk_id, source_version_id,
                            source_locator, ordinal, content_hash,
                            chunker_version, embedding, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, now())
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (
                        chunk.chunk_id,
                        chunk.source_version_id,
                        chunk.source_locator,
                        chunk.ordinal,
                        chunk.content_hash,
                        chunk.chunker_version,
                        vec_sql,
                    ))

                # 4. Insert release assignments
                if assignments:
                    for a in assignments:
                        cur.execute(f"""
                            INSERT INTO {self._schema}.release_assignments (
                                release_id, source_version_id,
                                audience_code, process_code, content_type_code,
                                source_role_code, approval_record,
                                reviewer_id, reviewed_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (
                            release.release_id,
                            a["source_version_id"],
                            a["audience_code"],
                            a["process_code"],
                            a["content_type_code"],
                            a.get("source_role_code", "supporting_authority"),
                            a.get("approval_record", ""),
                            a.get("reviewer_id", ""),
                            a.get("reviewed_at", ""),
                        ))

            conn.commit()

    def activate_release(self, release_id: str, activated_at: str) -> None:
        """Atomically swap the active-release pointer and emit usage outbox event.

        Idempotent: repeated calls for the same release_id are safe. If an
        activation outbox event already exists for this release, the active-release
        pointer is updated but no duplicate event is emitted.

        NOTE: This method writes a content-free system event to hr_policy.usage_events
        and hr_policy.usage_outbox. The connecting role must therefore hold INSERT on
        those tables in addition to the kb schema grants (hr_policy_writer or equivalent).
        The migration grants these to hr_policy_writer; ensure the publisher DSN connects
        as a role that inherits both hr_publisher and hr_policy_writer, or grant them
        explicitly to the publisher role.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Mark release activated
                cur.execute(f"""
                    UPDATE {self._schema}.releases
                    SET activated_at = %s
                    WHERE release_id = %s
                """, (activated_at, release_id))

                # Upsert singleton active-release pointer
                cur.execute(f"""
                    INSERT INTO {self._schema}.active_release (singleton, release_id, activated_at)
                    VALUES (TRUE, %s, %s)
                    ON CONFLICT (singleton) DO UPDATE
                        SET release_id   = EXCLUDED.release_id,
                            activated_at = EXCLUDED.activated_at
                """, (release_id, activated_at))

                # Emit content-free publication event to usage_outbox — idempotent.
                idem_key = f"activate_{release_id}"
                cur.execute("""
                    SELECT ue.usage_event_id
                    FROM hr_policy.usage_events ue
                    WHERE ue.customer_id = 'system'
                      AND ue.idempotency_key = %s
                    LIMIT 1
                """, (idem_key,))
                existing = cur.fetchone()

                if existing is None:
                    seq_ref = f"pub_{uuid.uuid4().hex}"
                    cur.execute("""
                        INSERT INTO hr_policy.usage_sequences
                            (usage_ref, customer_id, audience_code, process_code,
                             content_type_code, expires_at)
                        VALUES (%s, 'system', 'employer', 'hiring', 'question',
                                now() + interval '7 years')
                    """, (seq_ref,))

                    event_id = f"evt_{uuid.uuid4().hex}"
                    cur.execute("""
                        INSERT INTO hr_policy.usage_events
                            (usage_event_id, usage_ref, customer_id, idempotency_key,
                             event_type, process_code, content_type_code)
                        VALUES (%s, %s, 'system', %s, 'initial', 'hiring', 'question')
                        ON CONFLICT (customer_id, idempotency_key) DO NOTHING
                    """, (event_id, seq_ref, idem_key))

                    cur.execute("""
                        INSERT INTO hr_policy.usage_outbox (usage_event_id, created_at)
                        SELECT %s, now()
                        WHERE NOT EXISTS (
                            SELECT 1 FROM hr_policy.usage_outbox
                            WHERE usage_event_id = %s
                        )
                    """, (event_id, event_id))

            conn.commit()

    def rollback_release(self, release_id: str, activated_at: str) -> None:
        """Re-point active release at a prior validated release."""
        self.activate_release(release_id, activated_at)

    def set_source_state(
        self, source_version_id: str, state: str, activated_at: str
    ) -> None:
        """Record a global-state transition (suspend/withdraw) for a source version."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {self._schema}.source_versions
                    SET global_state = %s
                    WHERE source_version_id = %s
                """, (state, source_version_id))
            conn.commit()

    def search(
        self,
        vector: list[float],
        process: str,
        content_type: str,
        audience: str,
        limit: int = 5,
    ) -> list[dict]:
        """HNSW vector search over the active release filtered by assignment predicates.

        ``audience`` is a required trusted parameter ('employer' or 'employee').

        Joins active_release -> releases -> release_assignments -> chunks ->
        source_versions and filters by audience_code, process_code,
        content_type_code, and sv.global_state = 'valid'.

        Returns list of dicts: chunk_id, source_version_id, source_locator,
        content_hash, similarity, audience, process, content_type, source_role.
        Content text is never returned — callers must load it from the canonical
        origin if needed (ADR-0005 §11).
        """
        vec_sql = f"[{','.join(str(v) for v in vector)}]"
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT
                        c.chunk_id,
                        c.source_version_id,
                        c.source_locator,
                        c.content_hash,
                        1 - (c.embedding <=> %s::vector) AS similarity,
                        ra.audience_code     AS audience,
                        ra.process_code      AS process,
                        ra.content_type_code AS content_type,
                        ra.source_role_code  AS source_role
                    FROM {self._schema}.active_release ar
                    JOIN {self._schema}.releases r
                        ON r.release_id = ar.release_id
                    JOIN {self._schema}.release_assignments ra
                        ON ra.release_id = r.release_id
                    JOIN {self._schema}.chunks c
                        ON c.source_version_id = ra.source_version_id
                    JOIN {self._schema}.source_versions sv
                        ON sv.source_version_id = c.source_version_id
                    WHERE ra.audience_code     = %s
                      AND ra.process_code      = %s
                      AND ra.content_type_code = %s
                      AND sv.global_state      = 'valid'
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                """, (vec_sql, audience, process, content_type, vec_sql, limit))
                return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_release(row: dict) -> Release:
        return Release(
            release_id=row["release_id"],
            publication_run_id=row["publication_run_id"],
            product=row.get("product", "hr"),
            manifest_hash=row["manifest_hash"],
            policy_hash=row.get("policy_hash", ""),
            source_commit=row["source_commit"],
            chunker_version=row["chunker_version"],
            embedding_model_id=row["embedding_model_id"],
            build_status=row["build_status"],
            validation_status=row["validation_status"],
            activated_at=(
                row["activated_at"].isoformat() if row.get("activated_at") else None
            ),
            chunk_count=row.get("chunk_count", 0),
            assignment_count=row.get("assignment_count", 0),
        )
