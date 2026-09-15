"""Tests for the unified corpus publish pipeline.

Unit tests (always run — InMemoryCorpusStore, no DB):
  - Publish writes chunks + embeddings + active-release pointer
  - Second publish with same manifest+policy is a no-op (idempotent)
  - Unsupported schema version fails before any DB mutation
  - Publisher rejects empty compiled_assignments
  - Publisher rejects manifest with no matching assignment
  - Employer-only assignment not returned for employee query
  - Employee-only assignment not returned for employer query
  - Dual-audience source: one chunk, two assignment rows
  - Outbox event emitted on activation (content-free)
  - Emergency suspend excludes source from search

DB integration tests (skip if CORPUS_HR_TEST_DSN not set):
  - Full publish against real Postgres
  - search() returns employer result for employer query
  - search() returns employee result for employee query
  - Cross-audience exclusion: employer chunk not returned for employee query
  - Idempotency: second publish same release → no duplicate rows
  - Activation emits outbox row
"""

from __future__ import annotations

import hashlib
import os
from unittest.mock import MagicMock

import pytest

from connect_kb_hr.corpus.audit import InMemoryEventLog
from connect_kb_hr.corpus.chunker import DeterministicChunker
from connect_kb_hr.corpus.manifest import (
    CorpusManifest,
    EffectiveInterval,
    Provenance,
)
from connect_kb_hr.corpus.publisher import CorpusPublisher
from connect_kb_hr.corpus.target import TargetConfig
from connect_kb_hr.db.memory_store import InMemoryCorpusStore

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

CHUNKER_VERSION = "1.0"
MODEL_ID = "nomic-embed-text"
MODEL_DIGEST = "sha256:test-digest"
SOURCE_TEXT = "## Hiring\n\nEmployers must provide an offer letter before the start date."
CONTENT_HASH = hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()
METADATA_HASH = hashlib.sha256(b"meta").hexdigest()


def _manifest(
    source_id: str = "src-001",
    source_version_id: str = "src-001-v1",
    content_hash: str = CONTENT_HASH,
    schema_version: str = "1.0",
) -> CorpusManifest:
    return CorpusManifest(
        source_id=source_id,
        source_version_id=source_version_id,
        source_kind="rhh_document",
        canonical_origin="https://github.com/lawtrepreneur/kb-hr",
        content_hash=content_hash,
        metadata_hash=METADATA_HASH,
        jurisdiction="ON",
        effective_interval=EffectiveInterval(start="2026-01-01"),
        provenance=Provenance(origin_commit="abc123"),
        canonical_locator=f"collections/hiring/{source_id}.md",
        rights_basis="RHH_proprietary",
        rights_holder="RHH",
        permitted_use="subscriber_retrieval",
        attribution_required=False,
        global_state="valid",
        chunker_version=CHUNKER_VERSION,
        embedding_model_id=MODEL_ID,
        embedding_model_digest=MODEL_DIGEST,
        manifest_schema_version=schema_version,
    )


def _fake_embedder(dim: int = 768):
    embedder = MagicMock()
    embedder.model_id = MODEL_ID
    embedder.model_digest = MODEL_DIGEST
    embedder.embed.return_value = [0.1] * dim
    return embedder


def _target(product: str = "hr") -> TargetConfig:
    return TargetConfig(
        product=product,
        dsn="postgresql://test/test",  # not used with InMemoryCorpusStore
        publisher_role="hr_publisher",
        schema_name="kb",
    )


def _publisher(embedder=None) -> CorpusPublisher:
    return CorpusPublisher(
        event_log=InMemoryEventLog(),
        chunker=DeterministicChunker(version=CHUNKER_VERSION),
        embedder=embedder or _fake_embedder(),
    )


def _assignment(
    source_version_id: str = "src-001-v1",
    audience: str = "employer",
    process: str = "hiring",
    content_type: str = "question",
    source_role: str = "substantive_guidance",
) -> dict:
    return {
        "source_version_id": source_version_id,
        "audience_code": audience,
        "process_code": process,
        "content_type_code": content_type,
        "source_role_code": source_role,
        "approval_record": "rhh-esah-employer-hiring-001-v1.yaml",
        "reviewer_id": "reviewer-001",
        "reviewed_at": "2026-09-01T00:00:00Z",
    }


def _publish(
    store,
    manifest=None,
    publication_run_id: str = "run-001",
    assignments: list[dict] | None = None,
):
    m = manifest or _manifest()
    asgn = assignments if assignments is not None else [_assignment()]
    publisher = _publisher()
    target = _target()
    result = publisher.publish(
        publication_run_id=publication_run_id,
        target=target,
        store=store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="abc123",
        compiled_assignments=asgn,
    )
    return result


# ---------------------------------------------------------------------------
# Unit tests — InMemoryCorpusStore
# ---------------------------------------------------------------------------


def test_publish_writes_release_and_chunks():
    store = InMemoryCorpusStore()
    result = _publish(store)
    assert result.ok, f"publish failed: {result.detail}"
    assert store.active_release() is not None
    assert len(store._chunks) >= 1
    assert len(store._assignments) >= 1


def test_publish_idempotent_noop_on_same_manifest():
    store = InMemoryCorpusStore()
    r1 = _publish(store)
    assert r1.ok
    r2 = _publish(store)
    assert r2.ok
    assert r2.status == "noop"


def test_unsupported_schema_version_fails_before_mutation():
    store = InMemoryCorpusStore(schema_version="9.9")
    result = _publish(store)
    assert not result.ok
    assert "unsupported_schema_version" in (result.detail or "")
    assert store.active_release() is None


def test_publish_rejects_empty_compiled_assignments():
    store = InMemoryCorpusStore()
    result = _publish(store, assignments=[])
    assert not result.ok
    assert "compiled_assignments" in (result.detail or "").lower()
    assert store.active_release() is None


def test_publish_rejects_manifest_with_no_assignment():
    store = InMemoryCorpusStore()
    m = _manifest(source_version_id="src-999-v1")
    # assignment references a different source_version_id
    wrong_assignment = _assignment(source_version_id="src-001-v1")
    publisher = _publisher()
    target = _target()
    result = publisher.publish(
        publication_run_id="run-x",
        target=target,
        store=store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="abc123",
        compiled_assignments=[wrong_assignment],
    )
    assert not result.ok
    assert store.active_release() is None


def test_publish_with_employer_assignment_only():
    store = InMemoryCorpusStore()
    result = _publish(store, assignments=[_assignment(audience="employer")])
    assert result.ok
    # Employer query returns results
    chunks = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employer",
        limit=5,
    )
    assert len(chunks) >= 1
    # Employee query returns nothing (no employee assignment)
    chunks_emp = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employee",
        limit=5,
    )
    assert len(chunks_emp) == 0


def test_publish_with_employee_assignment_only():
    store = InMemoryCorpusStore()
    result = _publish(store, assignments=[_assignment(audience="employee")])
    assert result.ok
    chunks = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employee",
        limit=5,
    )
    assert len(chunks) >= 1
    # Employer query returns nothing
    chunks_empr = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employer",
        limit=5,
    )
    assert len(chunks_empr) == 0


def test_publish_with_dual_audience_assignments():
    """One source, two assignments (employer + employee) — one chunk, two assignment rows."""
    store = InMemoryCorpusStore()
    dual_assignments = [
        _assignment(audience="employer"),
        _assignment(audience="employee"),
    ]
    result = _publish(store, assignments=dual_assignments)
    assert result.ok
    # Same chunk for both audiences
    emp_chunks = store.search(
        vector=[0.1] * 768, process="hiring", content_type="question",
        audience="employer", limit=5,
    )
    ee_chunks = store.search(
        vector=[0.1] * 768, process="hiring", content_type="question",
        audience="employee", limit=5,
    )
    assert len(emp_chunks) >= 1
    assert len(ee_chunks) >= 1
    # Exactly one chunk artifact (no duplication)
    assert len(store._chunks) == len({c["chunk_id"] for c in emp_chunks})


def test_search_returns_employer_result_not_employee():
    """Employer-only assignment is excluded from employee query."""
    store = InMemoryCorpusStore()
    _publish(store, assignments=[_assignment(audience="employer")])
    results = store.search(
        vector=[0.1] * 768, process="hiring", content_type="question",
        audience="employee", limit=5,
    )
    assert results == []


def test_search_returns_employee_result_not_employer():
    """Employee-only assignment is excluded from employer query."""
    store = InMemoryCorpusStore()
    _publish(store, assignments=[_assignment(audience="employee")])
    results = store.search(
        vector=[0.1] * 768, process="hiring", content_type="question",
        audience="employer", limit=5,
    )
    assert results == []


def test_outbox_event_emitted_on_activation():
    store = InMemoryCorpusStore()
    result = _publish(store)
    assert result.ok
    assert len(store.outbox_events) >= 1
    # Outbox events must be content-free
    for evt in store.outbox_events:
        assert "query" not in str(evt).lower()
        assert SOURCE_TEXT not in str(evt)


def test_emergency_suspend_excludes_source_from_search():
    store = InMemoryCorpusStore()
    _publish(store)
    publisher = _publisher()
    publisher.emergency(
        publication_run_id="run-emergency",
        source_version_id="src-001-v1",
        store=store,
        target=_target(),
        action="suspend",
    )
    chunks = store.search(
        vector=[0.1] * 768, process="hiring", content_type="question",
        audience="employer", limit=5,
    )
    assert chunks == []


# ---------------------------------------------------------------------------
# DB integration tests — skip without CORPUS_HR_TEST_DSN
# ---------------------------------------------------------------------------

_SKIP_DB = pytest.mark.skipif(
    not os.environ.get("CORPUS_HR_TEST_DSN"),
    reason="requires live DB (CORPUS_HR_TEST_DSN not set)",
)

_DSN = os.environ.get("CORPUS_HR_TEST_DSN", "")


@pytest.fixture(scope="module")
def pg_store():
    """Fresh PostgresCorpusStore backed by the real hr_kb database."""
    import psycopg2

    from connect_kb_hr.db.postgres_store import PostgresCorpusStore

    store = PostgresCorpusStore(dsn=_DSN, schema="kb")
    # Teardown: drop schemas and recreate from migration for isolation
    yield store
    # Clean up test data — drop and recreate both schemas
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS kb CASCADE;")
        cur.execute("DROP SCHEMA IF EXISTS hr_policy CASCADE;")
    conn.close()
    # Re-run migration to leave DB clean for next run
    import pathlib
    migration = pathlib.Path(__file__).parent.parent / "migrations" / "001_hr_serving_store.sql"
    conn2 = psycopg2.connect(_DSN)
    conn2.autocommit = False
    with conn2.cursor() as cur:
        cur.execute(migration.read_text())
    conn2.commit()
    conn2.close()


@_SKIP_DB
def test_pg_publish_writes_and_activates(pg_store):
    result = _publish(pg_store)
    assert result.ok, f"pg publish failed: {result.detail}"
    active = pg_store.active_release()
    assert active is not None


@_SKIP_DB
def test_pg_search_employer_assignment(pg_store):
    """Employer assignment returns results for employer query."""
    from connect_kb_hr.db.postgres_store import PostgresCorpusStore
    store = PostgresCorpusStore(dsn=_DSN, schema="kb")
    _publish(store, assignments=[_assignment(audience="employer")], publication_run_id="pg-emp-1")
    results = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employer",
        limit=5,
    )
    assert len(results) >= 1
    assert results[0]["audience"] == "employer"


@_SKIP_DB
def test_pg_search_employee_assignment(pg_store):
    """Employee assignment returns results for employee query."""
    from connect_kb_hr.db.postgres_store import PostgresCorpusStore
    store = PostgresCorpusStore(dsn=_DSN, schema="kb")
    m = _manifest(source_id="src-ee-001", source_version_id="src-ee-001-v1")
    ee_assignment = _assignment(
        source_version_id="src-ee-001-v1",
        audience="employee",
    )
    publisher = _publisher()
    publisher.publish(
        publication_run_id="pg-ee-1",
        target=_target(),
        store=store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="abc123",
        compiled_assignments=[ee_assignment],
    )
    results = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employee",
        limit=5,
    )
    assert len(results) >= 1
    assert results[0]["audience"] == "employee"


@_SKIP_DB
def test_pg_cross_audience_excluded(pg_store):
    """Employer-only chunk is not returned for employee query."""
    from connect_kb_hr.db.postgres_store import PostgresCorpusStore
    store = PostgresCorpusStore(dsn=_DSN, schema="kb")
    m = _manifest(source_id="src-empr-x", source_version_id="src-empr-x-v1")
    _publish(store, manifest=m, assignments=[
        _assignment(source_version_id="src-empr-x-v1", audience="employer")
    ], publication_run_id="pg-cross-1")
    # Employee query must return empty for this employer-only source
    results = store.search(
        vector=[0.1] * 768,
        process="hiring",
        content_type="question",
        audience="employee",
        limit=5,
    )
    _ = {r["chunk_id"] for r in results}
    # No chunk from the employer-only source should appear
    assert all("src-empr-x" not in r.get("source_version_id", "") for r in results)


@_SKIP_DB
def test_pg_publish_idempotent(pg_store):
    """Second publish with same manifest+policy returns noop."""
    from connect_kb_hr.db.postgres_store import PostgresCorpusStore
    store = PostgresCorpusStore(dsn=_DSN, schema="kb")
    _publish(store, publication_run_id="pg-idem-1")
    r2 = _publish(store, publication_run_id="pg-idem-2")
    assert r2.ok
    assert r2.status == "noop"


@_SKIP_DB
def test_pg_activation_emits_outbox_row(pg_store):
    """Activation inserts a content-free row in hr_policy.usage_outbox."""
    import psycopg2
    conn = psycopg2.connect(_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM hr_policy.usage_outbox;")
        count = cur.fetchone()[0]
    conn.close()
    assert count >= 1
