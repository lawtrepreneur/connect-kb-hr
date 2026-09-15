"""Unified corpus/policy database foundation tests.

These tests verify the unified-corpus schema:
- kb schema: single corpus store with audience as an assignment predicate
- hr_policy schema: content-free usage tracking with audience as a data column
- Proper role-based access control
- HNSW index on chunks
- Singleton active_release
- Idempotent usage events
"""

import os
import pytest
import psycopg2
from psycopg2.extensions import connection, cursor

DB_NAME = "hr_kb"
DB_USER = "legalxmcp"
DB_PASS = "19a7567098f2c115ea920649cd58696f6cb02129"
DB_HOST = "127.0.0.1"
DB_PORT = 5435


def get_connection() -> "psycopg2.extensions.connection":
    """Return a connection to the test database."""
    dsn = os.environ.get("CORPUS_HR_TEST_DSN", "")
    if not dsn:
        pytest.skip("CORPUS_HR_TEST_DSN not set; skipping db tests")
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
    )


@pytest.fixture(scope="function")
def conn():
    """Database connection fixture."""
    conn = get_connection()
    yield conn
    conn.close()


def test_kb_schema_exists(conn: "psycopg2.extensions.connection") -> None:
    """Verify kb schema exists."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_tables WHERE schemaname = 'kb'")
    tables = cur.fetchall()
    assert len(tables) > 0, "kb schema should exist"
    cur.close()


def test_hr_policy_schema_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy schema exists."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_tables WHERE schemaname = 'hr_policy'")
    tables = cur.fetchall()
    assert len(tables) > 0, "hr_policy schema should exist"
    cur.close()


def test_kb_schema_version(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.schema_version exists and has version 1.0."""
    cur = conn.cursor()
    cur.execute("SELECT version, description FROM kb.schema_version LIMIT 1")
    row = cur.fetchone()
    assert row is not None, "kb.schema_version should have a row"
    assert row[0] == "1.0", f"Expected version 1.0, got {row[0]}"
    assert "unified corpus" in row[1].lower(), f"Description should mention unified corpus: {row[1]}"
    cur.close()


def test_kb_audience_table(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.audience has correct controlled values."""
    cur = conn.cursor()
    cur.execute("SELECT audience_code FROM kb.audience ORDER BY audience_code")
    rows = cur.fetchall()
    codes = [r[0] for r in rows]
    assert set(codes) == {"employer", "employee"}, f"Expected ['employer', 'employee'], got {codes}"
    cur.close()


def test_kb_process_table(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.process has correct controlled values."""
    cur = conn.cursor()
    cur.execute("SELECT process_code FROM kb.process ORDER BY process_code")
    rows = cur.fetchall()
    codes = [r[0] for r in rows]
    assert set(codes) == {
        "hiring",
        "onboarding",
        "performance",
        "leave",
        "discipline",
        "termination",
    }, f"Expected 6 process codes, got {codes}"
    cur.close()


def test_kb_content_type_table(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.content_type has correct controlled values."""
    cur = conn.cursor()
    cur.execute("SELECT content_type_code FROM kb.content_type ORDER BY content_type_code")
    rows = cur.fetchall()
    codes = [r[0] for r in rows]
    assert set(codes) == {"question", "checklist", "plan", "template"}, f"Expected 4 types, got {codes}"
    cur.close()


def test_kb_source_role_table(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.source_role has correct controlled values."""
    cur = conn.cursor()
    cur.execute("SELECT source_role_code FROM kb.source_role ORDER BY source_role_code")
    rows = cur.fetchall()
    codes = [r[0] for r in rows]
    assert set(codes) == {
        "supporting_authority",
        "substantive_guidance",
        "reusable_artifact",
    }, f"Expected 3 source roles, got {codes}"
    cur.close()


def test_kb_sources_table_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.sources table exists with correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'kb' AND table_name = 'sources'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "source_id",
        "source_kind",
        "canonical_origin",
        "rights_basis",
        "rights_holder",
        "permitted_use",
        "attribution_required",
        "created_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_kb_source_versions_table_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.source_versions table exists with correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'kb' AND table_name = 'source_versions'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "source_version_id",
        "source_id",
        "content_hash",
        "metadata_hash",
        "jurisdiction",
        "effective_start",
        "effective_end",
        "canonical_locator",
        "origin_commit",
        "global_state",
        "chunker_version",
        "embedding_model_id",
        "embedding_model_digest",
        "manifest_schema_version",
        "created_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_kb_chunks_table_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.chunks table exists with correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'kb' AND table_name = 'chunks'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "chunk_id",
        "source_version_id",
        "source_locator",
        "ordinal",
        "content_hash",
        "chunker_version",
        "embedding",
        "created_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_kb_releases_table_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.releases table exists with correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'kb' AND table_name = 'releases'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "release_id",
        "publication_run_id",
        "manifest_hash",
        "policy_hash",
        "source_commit",
        "chunker_version",
        "embedding_model_id",
        "build_status",
        "validation_status",
        "chunk_count",
        "assignment_count",
        "activated_at",
        "created_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_kb_active_release_table_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.active_release is a singleton."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM kb.active_release")
    count = cur.fetchone()[0]
    assert count <= 1, f"active_release singleton violated — got {count} rows"
    cur.close()


def test_kb_active_release_singleton_constraint(conn: psycopg2.extensions.connection) -> None:
    """Verify active_release has singleton CHECK constraint."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'kb.active_release'::regclass
          AND conname LIKE 'active_release_singleton%'
    """)
    rows = cur.fetchall()
    assert len(rows) >= 1, "active_release should have a singleton constraint"
    assert "singleton" in rows[0][0].lower(), f"Expected singleton constraint, got {rows[0][0]}"
    cur.close()


def test_kb_release_assignments_pk(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.release_assignments has correct primary key."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'kb.release_assignments'::regclass
          AND contype = 'p'
    """)
    pk_constraint = cur.fetchone()
    assert pk_constraint is not None, "release_assignments should have a primary key"
    cur.close()


def test_kb_hnsw_index_exists(conn: psycopg2.extensions.connection) -> None:
    """Verify HNSW index exists on kb.chunks.embedding."""
    cur = conn.cursor()
    cur.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'kb'
          AND tablename = 'chunks'
          AND indexname = 'chunks_embedding_hnsw_idx'
    """)
    row = cur.fetchone()
    assert row is not None, "HNSW index chunks_embedding_hnsw_idx should exist"
    cur.close()


def test_hr_policy_schema_version(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.schema_version exists and has version 1.0."""
    cur = conn.cursor()
    cur.execute("SELECT version, description FROM hr_policy.schema_version LIMIT 1")
    row = cur.fetchone()
    assert row is not None, "hr_policy.schema_version should have a row"
    assert row[0] == "1.0", f"Expected version 1.0, got {row[0]}"
    cur.close()


def test_hr_policy_usage_sequences(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_sequences has correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'hr_policy' AND table_name = 'usage_sequences'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "usage_ref",
        "customer_id",
        "audience_code",
        "process_code",
        "content_type_code",
        "created_at",
        "expires_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_hr_policy_usage_events(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_events has correct columns and idempotency constraint."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'hr_policy' AND table_name = 'usage_events'")
    cols = [r[0] for r in cur.fetchall()]
    expected = [
        "usage_event_id",
        "usage_ref",
        "customer_id",
        "idempotency_key",
        "event_type",
        "process_code",
        "content_type_code",
        "created_at",
    ]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_hr_policy_usage_outbox(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_outbox has correct columns."""
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'hr_policy' AND table_name = 'usage_outbox'")
    cols = [r[0] for r in cur.fetchall()]
    expected = ["outbox_id", "usage_event_id", "created_at", "delivered_at"]
    assert cols == expected, f"Expected columns {expected}, got {cols}"
    cur.close()


def test_hr_policy_event_type_check(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_events has CHECK constraint on event_type."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'hr_policy.usage_events'::regclass
          AND conname LIKE 'usage_events_event_type%'
    """)
    rows = cur.fetchall()
    assert len(rows) == 1, f"Expected event_type check constraint, got {rows}"
    assert "event_type" in rows[0][0].lower(), f"Expected event_type constraint, got {rows[0][0]}"
    cur.close()


def test_hr_policy_event_type_values(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_events CHECK constraint has correct values."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'hr_policy.usage_events'::regclass
          AND conname LIKE 'usage_events_event_type%'
    """)
    row = cur.fetchone()
    assert row is not None, "Should find event_type check constraint"
    constraint_def = row[2]
    assert "'initial'" in constraint_def, f"Expected 'initial' in constraint: {constraint_def}"
    assert "'clarification'" in constraint_def, f"Expected 'clarification' in constraint: {constraint_def}"
    assert "'refinement'" in constraint_def, f"Expected 'refinement' in constraint: {constraint_def}"
    assert "'completed'" in constraint_def, f"Expected 'completed' in constraint: {constraint_def}"
    cur.close()


def test_hr_policy_unique_idempotency(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy.usage_events has UNIQUE constraint on (customer_id, idempotency_key)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, contype FROM pg_constraint
        WHERE conrelid = 'hr_policy.usage_events'::regclass
          AND conname LIKE 'usage_events_idempotency%'
    """)
    rows = cur.fetchall()
    assert len(rows) == 1, "Should have idempotency uniqueness constraint"
    assert rows[0][1] == 'u', f"Expected unique constraint, got {rows[0][1]}"
    cur.close()


def test_hr_runtime_read_only_chunks(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_runtime cannot INSERT into kb.chunks."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_runtime'")
    grants = cur.fetchall()
    insert_grants = [g for g in grants if g[1] == "INSERT"]
    assert len(insert_grants) == 0, f"hr_runtime should not have INSERT on kb.chunks, got: {insert_grants}"
    cur.close()


def test_hr_publisher_can_insert_kb_sources(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_publisher can INSERT into kb.sources."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_publisher'")
    grants = cur.fetchall()
    insert_grants = [g for g in grants if g[3] == "INSERT" and g[1] == "kb"]
    assert len(insert_grants) >= 1, f"hr_publisher should have INSERT on kb tables, got: {insert_grants}"
    cur.close()


def test_hr_publisher_cannot_write_hr_policy(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_publisher has NO write access to hr_policy tables (per design — hr_policy_writer owns that)."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_publisher'")
    grants = cur.fetchall()
    insert_grants = [g for g in grants if g[3] == "INSERT" and g[1] == "hr_policy"]
    assert len(insert_grants) == 0, f"hr_publisher should NOT have INSERT on hr_policy (hr_policy_writer owns that), got: {insert_grants}"
    cur.close()


def test_hr_runtime_can_select_kb_chunks(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_runtime can SELECT from kb.chunks."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_runtime'")
    grants = cur.fetchall()
    select_grants = [g for g in grants if g[3] == "SELECT" and g[1] == "kb"]
    assert len(select_grants) >= 1, f"hr_runtime should have SELECT on kb tables, got: {select_grants}"
    cur.close()


def test_hr_policy_writer_can_write_hr_policy(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_policy_writer can INSERT/UPDATE on hr_policy tables."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_policy_writer'")
    grants = cur.fetchall()
    write_grants = [g for g in grants if g[3] in ("INSERT", "UPDATE") and g[1] == "hr_policy"]
    assert len(write_grants) >= 1, f"hr_policy_writer should have INSERT/UPDATE on hr_policy tables, got: {write_grants}"
    cur.close()


def test_hr_runtime_cannot_access_hr_policy(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_runtime does NOT have direct access to hr_policy tables (B4: per design)."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_runtime'")
    grants = cur.fetchall()
    policy_grants = [g for g in grants if g[1] == "hr_policy"]
    assert len(policy_grants) == 0, f"hr_runtime should NOT have direct access to hr_policy, got: {policy_grants}"
    cur.close()


def test_hr_publisher_can_write_kb(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_publisher can INSERT/UPDATE/DELETE on all kb tables."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_publisher'")
    grants = cur.fetchall()
    write_grants = [g for g in grants if g[3] in ("INSERT", "UPDATE", "DELETE") and g[1] == "kb"]
    assert len(write_grants) >= 1, f"hr_publisher should have full write on kb tables, got: {write_grants}"
    cur.close()


def test_hr_migrator_has_ddl_only(conn: psycopg2.extensions.connection) -> None:
    """Verify hr_migrator only has DDL privileges on kb schema."""
    cur = conn.cursor()
    cur.execute("SELECT grantee, table_schema, table_name, privilege_type FROM information_schema.role_table_grants WHERE grantee = 'hr_migrator'")
    grants = cur.fetchall()
    # hr_migrator should have USAGE on kb schema (DDL) but no INSERT/UPDATE/DELETE
    insert_grants = [g for g in grants if g[3] == "INSERT" and g[1] == "kb"]
    assert len(insert_grants) == 0, f"hr_migrator should not have INSERT on kb, got: {insert_grants}"
    cur.close()


def test_kb_chunks_unique_constraint(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.chunks has UNIQUE constraint on (source_version_id, ordinal)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'kb.chunks'::regclass
          AND contype = 'u'
    """)
    rows = cur.fetchall()
    unique_constraints = [r[2] for r in rows if "ordinal" in r[2].lower()]
    assert len(unique_constraints) >= 1, "kb.chunks should have UNIQUE constraint on (source_version_id, ordinal)"
    cur.close()


def test_kb_source_versions_unique_content(conn: psycopg2.extensions.connection) -> None:
    """Verify kb.source_versions has UNIQUE constraint on (source_id, content_hash)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT conname, conrelid::regclass, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'kb.source_versions'::regclass
          AND contype = 'u'
    """)
    rows = cur.fetchall()
    unique_constraints = [r[2] for r in rows if "content_hash" in r[2].lower()]
    assert len(unique_constraints) >= 1, "kb.source_versions should have UNIQUE constraint on (source_id, content_hash)"
    cur.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
