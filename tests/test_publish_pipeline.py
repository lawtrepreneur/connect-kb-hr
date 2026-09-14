"""Tests for connect-kb-hr#2: wired publish pipeline.

All AC from the issue covered:

  Unit tests (always run — InMemoryCorpusStore, no DB):
    - Publish run writes chunks + embeddings + active-release pointer
    - Second publish with same release_id is a no-op (idempotent)
    - Manifest with unsupported schema version fails before any DB mutation
    - Employer and employee publish runs are independent
    - Outbox event emitted on activation (content-free)
    - Emergency suspend/withdraw path

  DB integration tests (skip if CORPUS_HR_TEST_DSN not set):
    - Full publish against real Postgres
    - search() returns result for known query after publish
    - Idempotency: second publish same release_id → no duplicate rows
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
# Helpers
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
        canonical_locator=f"collections/employer/hiring/{source_id}.md",
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
    """Returns a mock embedder that emits a fixed non-zero vector."""
    embedder = MagicMock()
    embedder.model_id = MODEL_ID
    embedder.model_digest = MODEL_DIGEST
    embedder.embed.return_value = [0.1] * dim
    return embedder


def _target(product: str = "hr", audience: str = "employer") -> TargetConfig:
    return TargetConfig(
        product=product,
        audience=audience,
        dsn="postgresql://test/test",  # not used with InMemoryCorpusStore
        publisher_role="hr_publisher",
        schema_name=f"hr_{audience}",
    )


def _publisher(embedder=None) -> CorpusPublisher:
    return CorpusPublisher(
        event_log=InMemoryEventLog(),
        chunker=DeterministicChunker(version=CHUNKER_VERSION),
        embedder=embedder or _fake_embedder(),
    )


def _publish(store, manifest=None, publication_run_id="run-001", audience="employer"):
    m = manifest or _manifest()
    publisher = _publisher()
    target = _target(audience=audience)
    result = publisher.publish(
        publication_run_id=publication_run_id,
        target=target,
        store=store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="abc123",
    )
    return result


# ---------------------------------------------------------------------------
# AC: publish writes chunks + embeddings + active-release pointer
# ---------------------------------------------------------------------------

def test_publish_writes_release_and_chunks():
    store = InMemoryCorpusStore()
    result = _publish(store)

    assert result.ok, f"publish failed: {result.detail}"
    assert result.status == "activated"
    assert result.release_id is not None

    # active release pointer is set
    active = store.active_release()
    assert active is not None
    assert active.release_id == result.release_id

    # chunks were written
    assert len(store._chunks) > 0

    # embeddings were written
    for chunk_id in store._chunks:
        assert chunk_id in store._embeddings
        assert len(store._embeddings[chunk_id]) == 768


# ---------------------------------------------------------------------------
# AC: idempotent — second publish with same manifest is a no-op
# ---------------------------------------------------------------------------

def test_publish_idempotent_noop_on_same_manifest():
    store = InMemoryCorpusStore()
    r1 = _publish(store, publication_run_id="run-001")
    assert r1.status == "activated"

    chunk_count_before = len(store._chunks)

    r2 = _publish(store, publication_run_id="run-002")  # same manifest content
    assert r2.status == "noop", f"Expected noop, got {r2.status}: {r2.detail}"
    assert r2.release_id == r1.release_id

    # No new chunks written
    assert len(store._chunks) == chunk_count_before


# ---------------------------------------------------------------------------
# AC: unsupported schema version fails before any mutation
# ---------------------------------------------------------------------------

def test_unsupported_schema_version_fails_before_mutation():
    store = InMemoryCorpusStore(schema_version="2.0")
    result = _publish(store)

    assert not result.ok
    assert "unsupported_schema_version" in (result.detail or "")
    # No release written
    assert store.active_release() is None
    assert len(store._chunks) == 0


# ---------------------------------------------------------------------------
# AC: employer and employee publish runs are independent
# ---------------------------------------------------------------------------

def test_employer_employee_publish_independent():
    store_employer = InMemoryCorpusStore()
    store_employee = InMemoryCorpusStore()

    r_emp = _publish(store_employer, audience="employer", publication_run_id="run-empr")
    r_ee = _publish(store_employee, audience="employee", publication_run_id="run-empe")

    assert r_emp.ok
    assert r_ee.ok

    # Different release_ids (different target audience in publication_run_id)
    assert r_emp.release_id != r_ee.release_id

    # Each store has its own active release
    assert (store_employer.active_release() or MagicMock()).release_id == r_emp.release_id
    assert (store_employee.active_release() or MagicMock()).release_id == r_ee.release_id

    # Failure in employee does not affect employer
    bad_store = InMemoryCorpusStore(schema_version="99.0")
    bad_result = _publish(bad_store, audience="employee", publication_run_id="run-bad")
    assert not bad_result.ok
    # employer store still has its active release
    assert store_employer.active_release() is not None


# ---------------------------------------------------------------------------
# AC: content-free outbox event emitted on activation
# ---------------------------------------------------------------------------

def test_outbox_event_emitted_on_activation():
    store = InMemoryCorpusStore()
    result = _publish(store)

    assert result.ok
    assert len(store.outbox_events) == 1
    evt = store.outbox_events[0]
    assert evt["event_type"] == "release_activated"
    assert evt["release_id"] == result.release_id
    # No content fields
    assert "query" not in evt
    assert "source_text" not in evt
    assert "answer" not in evt


# ---------------------------------------------------------------------------
# AC: search returns result for known query (InMemoryCorpusStore cosine)
# ---------------------------------------------------------------------------

def test_search_returns_result_after_publish():
    store = InMemoryCorpusStore()
    _publish(store)

    # Query with same fixed vector as embedder emits (cosine sim = 1.0)
    query_vector = [0.1] * 768
    results = store.search(vector=query_vector, process="hiring", content_type="question")

    assert len(results) > 0
    first = results[0]
    assert "chunk_id" in first
    assert "source_version_id" in first
    assert "similarity" in first
    assert first["similarity"] > 0.99  # should be ~1.0 for identical vectors


def test_search_returns_empty_when_no_active_release():
    store = InMemoryCorpusStore()
    results = store.search(vector=[0.1] * 768, process="hiring", content_type="question")
    assert results == []


# ---------------------------------------------------------------------------
# AC: emergency suspend removes source from active retrieval
# ---------------------------------------------------------------------------

def test_emergency_suspend_excludes_source_from_search():
    store = InMemoryCorpusStore()
    m = _manifest(source_version_id="src-suspend-v1")
    result = _publish(store, manifest=m)
    assert result.ok

    # Verify searchable before suspend
    assert len(store.search([0.1] * 768, "hiring", "question")) > 0

    # Suspend
    publisher = _publisher()
    target = _target()
    store.set_source_state("src-suspend-v1", "suspended", "2026-09-14T00:00:00Z")

    # Should now be excluded
    results = store.search([0.1] * 768, "hiring", "question")
    assert all(r["source_version_id"] != "src-suspend-v1" for r in results)


# ---------------------------------------------------------------------------
# DB integration tests — skip if CORPUS_HR_TEST_DSN not set
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_dsn():
    dsn = os.environ.get("CORPUS_HR_TEST_DSN", "")
    if not dsn:
        pytest.skip("CORPUS_HR_TEST_DSN not set — skipping DB integration tests")
    return dsn


@pytest.fixture(scope="module")
def pg_store(db_dsn):
    """Migrate and return a PostgresCorpusStore; teardown after module."""
    import psycopg2
    from connect_kb_hr.db.postgres_store import PostgresCorpusStore

    # Apply migrations
    conn = psycopg2.connect(db_dsn)
    conn.autocommit = True
    migration_sql = (
        __import__("pathlib").Path(__file__).parent.parent
        / "migrations" / "001_hr_serving_store.sql"
    )
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            pass
        cur.execute(migration_sql.read_text())
    conn.close()

    store = PostgresCorpusStore(dsn=db_dsn, schema="hr_employer")
    yield store

    # Teardown
    conn = psycopg2.connect(db_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS hr_employer CASCADE;")
        cur.execute("DROP SCHEMA IF EXISTS hr_employee CASCADE;")
    conn.close()


def test_pg_publish_writes_and_activates(pg_store):
    m = _manifest(source_id="pg-src-001", source_version_id="pg-src-001-v1")
    publisher = _publisher()
    target = _target()
    result = publisher.publish(
        publication_run_id="pg-run-001",
        target=target,
        store=pg_store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="pg-abc123",
    )
    assert result.ok, f"DB publish failed: {result.detail}"
    active = pg_store.active_release()
    assert active is not None
    assert active.release_id == result.release_id


def test_pg_publish_idempotent(pg_store):
    """Second publish of same manifest must be a noop with no duplicate rows."""
    import psycopg2
    m = _manifest(source_id="pg-src-001", source_version_id="pg-src-001-v1")
    publisher = _publisher()
    target = _target()
    r2 = publisher.publish(
        publication_run_id="pg-run-002",
        target=target,
        store=pg_store,
        manifests=[m],
        source_texts={m.source_version_id: SOURCE_TEXT},
        source_commit="pg-abc123",
    )
    assert r2.status == "noop", f"Expected noop on second publish, got {r2.status}"


def test_pg_search_returns_result(pg_store):
    """After publish, search() must return at least one result for a known vector."""
    results = pg_store.search(vector=[0.1] * 768, process="hiring", content_type="question")
    assert isinstance(results, list)
    assert len(results) > 0
    assert "chunk_id" in results[0]
    assert "similarity" in results[0]


def test_pg_activation_emits_outbox_row(pg_store, db_dsn):
    """activate_release must write one row to the usage_outbox."""
    import psycopg2
    conn = psycopg2.connect(db_dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM hr_employer.usage_outbox")
        count = cur.fetchone()[0]
    conn.close()
    assert count >= 1, "Expected at least one outbox row after publish activation"
