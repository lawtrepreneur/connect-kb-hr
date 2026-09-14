"""PostgreSQL CorpusStore adapter — connect-kb-hr#1 / launch-mcp#28.

Implements the CorpusStore protocol defined in publisher.py against the
hr_employer / hr_employee schemas created by migrations/001_hr_serving_store.sql.

Credentials come from environment variables only (CORPUS_HR_EMPLOYER_DSN,
CORPUS_HR_EMPLOYEE_DSN). DSNs are never logged or exposed in repr output.

The store is instantiated once per publication run; it does not hold a
long-lived connection pool (the publisher is a batch process, not a server).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Mapping, Sequence

import psycopg2
import psycopg2.extras

from connect_kb_hr.corpus.chunker import Chunk
from connect_kb_hr.corpus.manifest import CorpusManifest
from connect_kb_hr.corpus.publisher import CorpusStore
from connect_kb_hr.corpus.release import Release

# Schema version this adapter was written against.
# The publisher gates on this before any mutation.
SUPPORTED_SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgresCorpusStore:
    """PostgreSQL implementation of CorpusStore for one audience schema.

    ``dsn`` is taken from environment variables; pass ``audience`` as
    ``'employer'`` or ``'employee'`` to select the correct schema.
    Never log, print, or repr the dsn.
    """

    def __init__(self, dsn: str, schema: str) -> None:
        if not dsn:
            raise ValueError("dsn is required (never logged)")
        if schema not in ("hr_employer", "hr_employee"):
            raise ValueError(f"schema must be hr_employer or hr_employee, got {schema!r}")
        self._dsn = dsn
        self._schema = schema

    def __repr__(self) -> str:
        return f"PostgresCorpusStore(schema={self._schema!r}, dsn=<redacted>)"

    @classmethod
    def from_env(cls, audience: str) -> "PostgresCorpusStore":
        """Build a store from CORPUS_HR_{AUDIENCE}_DSN env var."""
        key = f"CORPUS_HR_{audience.upper()}_DSN"
        dsn = os.environ.get(key, "")
        if not dsn:
            raise ValueError(f"{key} is required (no default)")
        schema = f"hr_{audience}"
        return cls(dsn=dsn, schema=schema)

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
                    raise RuntimeError(f"No schema_version row found in {self._schema}")
                return row[0]

    def active_release(self) -> Release | None:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT r.*,
                           split_part('{self._schema}', '_', 2) AS audience
                    FROM {self._schema}.releases r
                    JOIN {self._schema}.active_release ar ON ar.release_id = r.release_id
                """)
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_release(row)

    def validated_releases(self) -> list[Release]:
        audience = self._schema.split("_", 1)[1]  # hr_employer -> employer
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT *, %s AS audience
                    FROM {self._schema}.releases
                    WHERE validation_status = 'passed'
                    ORDER BY created_at DESC
                """, (audience,))
                return [self._row_to_release(row) for row in cur.fetchall()]

    def write_release(
        self,
        release: Release,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, list[float]],
        manifests: Sequence[CorpusManifest] | None = None,
    ) -> None:
        """Write release + source projections + chunks + embeddings atomically.

        ``manifests`` is optional but required for sources/source_versions to
        be written. Without it, FK constraints on chunks will fail if the
        source_version rows do not already exist.

        Idempotent: ON CONFLICT DO NOTHING throughout.
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
                        release_id, publication_run_id, manifest_hash, source_commit,
                        chunker_version, embedding_model_id, build_status, validation_status,
                        chunk_count, activated_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (release_id) DO NOTHING
                """, (
                    release.release_id,
                    release.publication_run_id,
                    release.manifest_hash,
                    release.source_commit,
                    release.chunker_version,
                    release.embedding_model_id,
                    release.build_status,
                    release.validation_status,
                    release.chunk_count,
                    release.activated_at,
                ))

                # 3. Insert chunks + embeddings
                for chunk in chunks:
                    vec = embeddings.get(chunk.chunk_id)
                    vec_sql = (
                        f"[{','.join(str(v) for v in vec)}]" if vec is not None else None
                    )
                    cur.execute(f"""
                        INSERT INTO {self._schema}.chunks (
                            chunk_id, release_id, source_version_id,
                            source_locator, ordinal, content_hash,
                            chunker_version, embedding, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s,
                                  %s::vector, now())
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (
                        chunk.chunk_id,
                        release.release_id,
                        chunk.source_version_id,
                        chunk.source_locator,
                        chunk.ordinal,
                        chunk.content_hash,
                        chunk.chunker_version,
                        vec_sql,
                    ))

            conn.commit()

    def activate_release(self, release_id: str, activated_at: str) -> None:
        """Atomically swap the active-release pointer and emit usage outbox event.

        Idempotent: repeated calls for the same release_id are safe. If an
        activation outbox event already exists for this release, the active-release
        pointer is updated but no duplicate event is emitted.
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
                # Check whether an activation event for this release already exists
                # before inserting to avoid duplicate sequences and events.
                idem_key = f"activate_{release_id}"
                cur.execute(f"""
                    SELECT ue.usage_event_id
                    FROM {self._schema}.usage_events ue
                    WHERE ue.customer_id = 'system'
                      AND ue.idempotency_key = %s
                    LIMIT 1
                """, (idem_key,))
                existing = cur.fetchone()

                if existing is None:
                    audience = self._schema.split("_", 1)[1]
                    seq_ref = f"pub_{uuid.uuid4().hex}"
                    cur.execute(f"""
                        INSERT INTO {self._schema}.usage_sequences
                            (usage_ref, customer_id, audience, process, content_type, expires_at)
                        VALUES (%s, 'system', %s, 'publication', 'event',
                                now() + interval '7 years')
                    """, (seq_ref, audience))

                    event_id = f"evt_{uuid.uuid4().hex}"
                    cur.execute(f"""
                        INSERT INTO {self._schema}.usage_events
                            (usage_event_id, usage_ref, customer_id, idempotency_key,
                             event_type, process, content_type)
                        VALUES (%s, %s, 'system', %s, 'initial', 'publication', 'event')
                        ON CONFLICT (customer_id, idempotency_key) DO NOTHING
                    """, (event_id, seq_ref, idem_key))

                    cur.execute(f"""
                        INSERT INTO {self._schema}.usage_outbox (usage_event_id, created_at)
                        SELECT %s, now()
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {self._schema}.usage_outbox
                            WHERE usage_event_id = %s
                        )
                    """, (event_id, event_id))

            conn.commit()

    def rollback_release(self, release_id: str, activated_at: str) -> None:
        """Re-point active release at a prior validated release."""
        self.activate_release(release_id, activated_at)

    def set_source_state(self, source_version_id: str, state: str, activated_at: str) -> None:
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
        limit: int = 5,
    ) -> list[dict]:
        """HNSW vector search over the active release.

        Returns list of dicts with: source_version_id, source_locator,
        content_hash, similarity. Content text is never returned — callers
        must load it from the canonical origin if needed (ADR-0005 §11).

        Skips chunks from suspended/withdrawn source versions.

        NOTE — process/content_type filtering (tech debt):
        The ``process`` and ``content_type`` parameters are accepted but not
        yet applied in the SQL query. Filtering requires a release-policy table
        that maps (release_id, source_version_id, process, content_type) tuples
        — populated from the compiled approval artifact produced by kb-hr's
        compile_approvals.py compiler. Until that table exists and is populated
        by the publisher, all active-release chunks are returned regardless of
        process/content_type. Tracked in connect-kb-hr#2.
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
                        1 - (c.embedding <=> %s::vector) AS similarity
                    FROM {self._schema}.chunks c
                    JOIN {self._schema}.active_release ar ON TRUE
                    JOIN {self._schema}.source_versions sv
                        ON sv.source_version_id = c.source_version_id
                    WHERE c.release_id = ar.release_id
                      AND sv.global_state = 'valid'
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                """, (vec_sql, vec_sql, limit))
                return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_release(row: dict) -> Release:
        return Release(
            release_id=row["release_id"],
            publication_run_id=row["publication_run_id"],
            product="hr",
            audience=row.get("audience", ""),
            manifest_hash=row["manifest_hash"],
            source_commit=row["source_commit"],
            chunker_version=row["chunker_version"],
            embedding_model_id=row["embedding_model_id"],
            build_status=row["build_status"],
            validation_status=row["validation_status"],
            activated_at=(
                row["activated_at"].isoformat() if row.get("activated_at") else None
            ),
            chunk_count=row.get("chunk_count", 0),
        )
