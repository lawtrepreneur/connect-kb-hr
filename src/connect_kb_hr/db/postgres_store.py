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
from typing import Mapping, Sequence

import psycopg2
import psycopg2.extras

from connect_kb_hr.corpus.chunker import Chunk
from connect_kb_hr.corpus.publisher import CorpusStore
from connect_kb_hr.corpus.release import Release, ReleaseBuilder

# Schema version this adapter was written against.
# The publisher gates on this before any mutation.
SUPPORTED_SCHEMA_VERSION = "1.0"


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
                    f"SELECT version FROM {self._schema}.schema_version ORDER BY applied_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"No schema_version row found in {self._schema}")
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
                    SELECT * FROM {self._schema}.releases
                    WHERE validation_status = 'passed'
                    ORDER BY created_at DESC
                """)
                return [self._row_to_release(row) for row in cur.fetchall()]

    def write_release(
        self,
        release: Release,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, list[float]],
    ) -> None:
        """Write release + chunks + embeddings atomically (not yet activated)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Insert release
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

                # Insert chunks + embeddings
                for chunk in chunks:
                    vec = embeddings.get(chunk.chunk_id)
                    # psycopg2 pgvector adapter expects list -> cast to string '[...]'
                    vec_sql = f"[{','.join(str(v) for v in vec)}]" if vec else None
                    cur.execute(f"""
                        INSERT INTO {self._schema}.chunks (
                            chunk_id, release_id, source_version_id,
                            source_locator, ordinal, content_hash,
                            chunker_version, embedding, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, now())
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
        """Atomically swap the active-release pointer."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Mark release activated
                cur.execute(f"""
                    UPDATE {self._schema}.releases
                    SET activated_at = %s
                    WHERE release_id = %s
                """, (activated_at, release_id))
                # Upsert the singleton active-release row
                cur.execute(f"""
                    INSERT INTO {self._schema}.active_release (singleton, release_id, activated_at)
                    VALUES (TRUE, %s, %s)
                    ON CONFLICT (singleton) DO UPDATE
                        SET release_id = EXCLUDED.release_id,
                            activated_at = EXCLUDED.activated_at
                """, (release_id, activated_at))
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
            activated_at=row["activated_at"].isoformat() if row.get("activated_at") else None,
            chunk_count=row.get("chunk_count", 0),
        )
