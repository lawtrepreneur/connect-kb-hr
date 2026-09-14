-- Migration 001: HR serving-store foundation
-- connect-kb-hr#1 / launch-mcp#28
--
-- Applies to: hr_employer and hr_employee schemas (run twice, once per audience)
-- Roles created: hr_publisher, hr_employer_reader, hr_employee_reader, hr_policy_runtime
-- Schema version tag: 1.0
--
-- Prerequisites:
--   - PostgreSQL >= 14 with pgvector extension available
--   - Run as a superuser the first time (to create roles and grant schema ownership)
--   - Set :schema to hr_employer or hr_employee before running
--
-- Rollback: see 001_rollback.sql

-- ---------------------------------------------------------------------------
-- 0. Roles (idempotent — skip if already exist)
-- ---------------------------------------------------------------------------

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_publisher') THEN
    CREATE ROLE hr_publisher NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_employer_reader') THEN
    CREATE ROLE hr_employer_reader NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_employee_reader') THEN
    CREATE ROLE hr_employee_reader NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_policy_runtime') THEN
    CREATE ROLE hr_policy_runtime NOLOGIN;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 1. hr_employer schema
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS hr_employer;
GRANT USAGE ON SCHEMA hr_employer TO hr_publisher, hr_employer_reader, hr_policy_runtime;
-- hr_employee_reader deliberately NOT granted hr_employer usage
-- hr_policy_runtime gets SCHEMA USAGE only (no corpus table access — usage tables only)

-- Enable pgvector in this database (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS hr_employer.schema_version (
    version         TEXT        NOT NULL,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    description     TEXT        NOT NULL
);
INSERT INTO hr_employer.schema_version (version, description)
VALUES ('1.0', 'initial: release tables, HNSW index, active-release pointer, usage outbox')
ON CONFLICT DO NOTHING;

-- Sources: stable source identity (content-free metadata only)
CREATE TABLE IF NOT EXISTS hr_employer.sources (
    source_id           TEXT        NOT NULL,
    source_kind         TEXT        NOT NULL CHECK (source_kind IN ('clause','casebrief','rhh_document','legislation')),
    canonical_origin    TEXT        NOT NULL,
    rights_basis        TEXT        NOT NULL,
    rights_holder       TEXT        NOT NULL,
    permitted_use       TEXT        NOT NULL,
    attribution_required BOOLEAN    NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT sources_pk PRIMARY KEY (source_id)
);

-- Source versions: immutable, content-free metadata + hashes
CREATE TABLE IF NOT EXISTS hr_employer.source_versions (
    source_version_id       TEXT        NOT NULL,
    source_id               TEXT        NOT NULL REFERENCES hr_employer.sources(source_id),
    content_hash            TEXT        NOT NULL,
    metadata_hash           TEXT        NOT NULL,
    jurisdiction            TEXT        NOT NULL,
    effective_start         DATE        NOT NULL,
    effective_end           DATE,
    canonical_locator       TEXT        NOT NULL,
    origin_commit           TEXT        NOT NULL,
    global_state            TEXT        NOT NULL CHECK (global_state IN ('valid','suspended','withdrawn','superseded')),
    chunker_version         TEXT        NOT NULL,
    embedding_model_id      TEXT        NOT NULL,
    embedding_model_digest  TEXT        NOT NULL,
    manifest_schema_version TEXT        NOT NULL DEFAULT '1.0',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT source_versions_pk PRIMARY KEY (source_version_id)
);

CREATE INDEX IF NOT EXISTS source_versions_source_id_idx
    ON hr_employer.source_versions (source_id);
CREATE INDEX IF NOT EXISTS source_versions_global_state_idx
    ON hr_employer.source_versions (global_state);

-- Releases: immutable publication units
CREATE TABLE IF NOT EXISTS hr_employer.releases (
    release_id              TEXT        NOT NULL,
    publication_run_id      TEXT        NOT NULL,
    manifest_hash           TEXT        NOT NULL,
    source_commit           TEXT        NOT NULL,
    chunker_version         TEXT        NOT NULL,
    embedding_model_id      TEXT        NOT NULL,
    build_status            TEXT        NOT NULL CHECK (build_status IN ('building','built','failed')),
    validation_status       TEXT        NOT NULL CHECK (validation_status IN ('pending','passed','failed')),
    chunk_count             INTEGER     NOT NULL DEFAULT 0,
    activated_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT releases_pk PRIMARY KEY (release_id)
);

CREATE INDEX IF NOT EXISTS releases_validation_status_idx
    ON hr_employer.releases (validation_status, created_at DESC);

-- Active-release pointer: exactly one row; swap atomically on publish/rollback
CREATE TABLE IF NOT EXISTS hr_employer.active_release (
    singleton               BOOLEAN     NOT NULL DEFAULT TRUE CHECK (singleton),
    release_id              TEXT        NOT NULL REFERENCES hr_employer.releases(release_id),
    activated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT active_release_singleton PRIMARY KEY (singleton)
);

-- Chunks: immutable, derived from source versions (content stored, never logged)
CREATE TABLE IF NOT EXISTS hr_employer.chunks (
    chunk_id            TEXT        NOT NULL,
    release_id          TEXT        NOT NULL REFERENCES hr_employer.releases(release_id),
    source_version_id   TEXT        NOT NULL REFERENCES hr_employer.source_versions(source_version_id),
    source_locator      TEXT        NOT NULL,
    ordinal             INTEGER     NOT NULL,
    content_hash        TEXT        NOT NULL,
    chunker_version     TEXT        NOT NULL,
    embedding           vector(768),  -- dimension matches nomic-embed-text; adjust per model
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_pk PRIMARY KEY (chunk_id),
    CONSTRAINT chunks_unique_ordinal UNIQUE (source_version_id, ordinal, release_id)
);

-- HNSW index for semantic search (pgvector cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON hr_employer.chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- FTS index on content_hash (for audit/provenance — chunk content itself is not logged)
CREATE INDEX IF NOT EXISTS chunks_release_source_idx
    ON hr_employer.chunks (release_id, source_version_id);

-- ---------------------------------------------------------------------------
-- Usage outbox (content-free) — shared by hr_policy_runtime
-- All fields: identifiers, category tags, timestamps only
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hr_employer.usage_sequences (
    usage_ref           TEXT        NOT NULL,   -- opaque, high-entropy
    customer_id         TEXT        NOT NULL,
    audience            TEXT        NOT NULL DEFAULT 'employer',
    process             TEXT        NOT NULL,
    content_type        TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ NOT NULL,
    CONSTRAINT usage_sequences_pk PRIMARY KEY (usage_ref)
);

CREATE TABLE IF NOT EXISTS hr_employer.usage_events (
    usage_event_id      TEXT        NOT NULL,
    usage_ref           TEXT        NOT NULL REFERENCES hr_employer.usage_sequences(usage_ref),
    customer_id         TEXT        NOT NULL,
    idempotency_key     TEXT        NOT NULL,
    event_type          TEXT        NOT NULL CHECK (event_type IN ('initial','clarification','refinement','completed')),
    process             TEXT        NOT NULL,
    content_type        TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT usage_events_pk PRIMARY KEY (usage_event_id),
    CONSTRAINT usage_events_idempotency UNIQUE (customer_id, idempotency_key)
);

-- Outbox for reliable delivery (write-ahead, mark delivered)
CREATE TABLE IF NOT EXISTS hr_employer.usage_outbox (
    outbox_id           BIGSERIAL   PRIMARY KEY,
    usage_event_id      TEXT        NOT NULL REFERENCES hr_employer.usage_events(usage_event_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at        TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- 2. hr_employee schema (identical structure, independent roles)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS hr_employee;
GRANT USAGE ON SCHEMA hr_employee TO hr_publisher, hr_employee_reader, hr_policy_runtime;
-- hr_employer_reader deliberately NOT granted hr_employee usage
-- hr_policy_runtime gets SCHEMA USAGE only (no corpus table access — usage tables only)

CREATE TABLE IF NOT EXISTS hr_employee.schema_version (
    version         TEXT        NOT NULL,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    description     TEXT        NOT NULL
);
INSERT INTO hr_employee.schema_version (version, description)
VALUES ('1.0', 'initial: release tables, HNSW index, active-release pointer, usage outbox')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS hr_employee.sources (
    source_id           TEXT        NOT NULL,
    source_kind         TEXT        NOT NULL CHECK (source_kind IN ('clause','casebrief','rhh_document','legislation')),
    canonical_origin    TEXT        NOT NULL,
    rights_basis        TEXT        NOT NULL,
    rights_holder       TEXT        NOT NULL,
    permitted_use       TEXT        NOT NULL,
    attribution_required BOOLEAN    NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT sources_pk PRIMARY KEY (source_id)
);

CREATE TABLE IF NOT EXISTS hr_employee.source_versions (
    source_version_id       TEXT        NOT NULL,
    source_id               TEXT        NOT NULL REFERENCES hr_employee.sources(source_id),
    content_hash            TEXT        NOT NULL,
    metadata_hash           TEXT        NOT NULL,
    jurisdiction            TEXT        NOT NULL,
    effective_start         DATE        NOT NULL,
    effective_end           DATE,
    canonical_locator       TEXT        NOT NULL,
    origin_commit           TEXT        NOT NULL,
    global_state            TEXT        NOT NULL CHECK (global_state IN ('valid','suspended','withdrawn','superseded')),
    chunker_version         TEXT        NOT NULL,
    embedding_model_id      TEXT        NOT NULL,
    embedding_model_digest  TEXT        NOT NULL,
    manifest_schema_version TEXT        NOT NULL DEFAULT '1.0',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT source_versions_pk PRIMARY KEY (source_version_id)
);

CREATE INDEX IF NOT EXISTS source_versions_source_id_idx
    ON hr_employee.source_versions (source_id);
CREATE INDEX IF NOT EXISTS source_versions_global_state_idx
    ON hr_employee.source_versions (global_state);

CREATE TABLE IF NOT EXISTS hr_employee.releases (
    release_id              TEXT        NOT NULL,
    publication_run_id      TEXT        NOT NULL,
    manifest_hash           TEXT        NOT NULL,
    source_commit           TEXT        NOT NULL,
    chunker_version         TEXT        NOT NULL,
    embedding_model_id      TEXT        NOT NULL,
    build_status            TEXT        NOT NULL CHECK (build_status IN ('building','built','failed')),
    validation_status       TEXT        NOT NULL CHECK (validation_status IN ('pending','passed','failed')),
    chunk_count             INTEGER     NOT NULL DEFAULT 0,
    activated_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT releases_pk PRIMARY KEY (release_id)
);

CREATE INDEX IF NOT EXISTS releases_validation_status_idx
    ON hr_employee.releases (validation_status, created_at DESC);

CREATE TABLE IF NOT EXISTS hr_employee.active_release (
    singleton               BOOLEAN     NOT NULL DEFAULT TRUE CHECK (singleton),
    release_id              TEXT        NOT NULL REFERENCES hr_employee.releases(release_id),
    activated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT active_release_singleton PRIMARY KEY (singleton)
);

CREATE TABLE IF NOT EXISTS hr_employee.chunks (
    chunk_id            TEXT        NOT NULL,
    release_id          TEXT        NOT NULL REFERENCES hr_employee.releases(release_id),
    source_version_id   TEXT        NOT NULL REFERENCES hr_employee.source_versions(source_version_id),
    source_locator      TEXT        NOT NULL,
    ordinal             INTEGER     NOT NULL,
    content_hash        TEXT        NOT NULL,
    chunker_version     TEXT        NOT NULL,
    embedding           vector(768),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_pk PRIMARY KEY (chunk_id),
    CONSTRAINT chunks_unique_ordinal UNIQUE (source_version_id, ordinal, release_id)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON hr_employee.chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_release_source_idx
    ON hr_employee.chunks (release_id, source_version_id);

CREATE TABLE IF NOT EXISTS hr_employee.usage_sequences (
    usage_ref           TEXT        NOT NULL,
    customer_id         TEXT        NOT NULL,
    audience            TEXT        NOT NULL DEFAULT 'employee',
    process             TEXT        NOT NULL,
    content_type        TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ NOT NULL,
    CONSTRAINT usage_sequences_pk PRIMARY KEY (usage_ref)
);

CREATE TABLE IF NOT EXISTS hr_employee.usage_events (
    usage_event_id      TEXT        NOT NULL,
    usage_ref           TEXT        NOT NULL REFERENCES hr_employee.usage_sequences(usage_ref),
    customer_id         TEXT        NOT NULL,
    idempotency_key     TEXT        NOT NULL,
    event_type          TEXT        NOT NULL CHECK (event_type IN ('initial','clarification','refinement','completed')),
    process             TEXT        NOT NULL,
    content_type        TEXT        NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT usage_events_pk PRIMARY KEY (usage_event_id),
    CONSTRAINT usage_events_idempotency UNIQUE (customer_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS hr_employee.usage_outbox (
    outbox_id           BIGSERIAL   PRIMARY KEY,
    usage_event_id      TEXT        NOT NULL REFERENCES hr_employee.usage_events(usage_event_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at        TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- 3. Grant table-level permissions per role
-- ---------------------------------------------------------------------------

-- hr_publisher: full write on both schemas
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA hr_employer TO hr_publisher;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA hr_employer TO hr_publisher;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA hr_employee TO hr_publisher;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA hr_employee TO hr_publisher;

-- hr_employer_reader: read active releases and safe source metadata; no chunks content (embedding is safe)
GRANT SELECT ON hr_employer.releases        TO hr_employer_reader;
GRANT SELECT ON hr_employer.active_release  TO hr_employer_reader;
GRANT SELECT ON hr_employer.source_versions TO hr_employer_reader;
GRANT SELECT ON hr_employer.sources         TO hr_employer_reader;
GRANT SELECT ON hr_employer.chunks          TO hr_employer_reader;  -- embeddings for retrieval
GRANT SELECT ON hr_employer.schema_version  TO hr_employer_reader;
-- hr_employer_reader cannot see hr_employee schema (no USAGE grant)

-- hr_employee_reader: symmetric
GRANT SELECT ON hr_employee.releases        TO hr_employee_reader;
GRANT SELECT ON hr_employee.active_release  TO hr_employee_reader;
GRANT SELECT ON hr_employee.source_versions TO hr_employee_reader;
GRANT SELECT ON hr_employee.sources         TO hr_employee_reader;
GRANT SELECT ON hr_employee.chunks          TO hr_employee_reader;
GRANT SELECT ON hr_employee.schema_version  TO hr_employee_reader;

-- hr_policy_runtime: usage outbox write + sequence read; no corpus chunk access
GRANT SELECT, INSERT ON hr_employer.usage_sequences TO hr_policy_runtime;
GRANT SELECT, INSERT ON hr_employer.usage_events    TO hr_policy_runtime;
GRANT SELECT, INSERT, UPDATE ON hr_employer.usage_outbox TO hr_policy_runtime;
GRANT USAGE, SELECT ON SEQUENCE hr_employer.usage_outbox_outbox_id_seq TO hr_policy_runtime;
GRANT SELECT, INSERT ON hr_employee.usage_sequences TO hr_policy_runtime;
GRANT SELECT, INSERT ON hr_employee.usage_events    TO hr_policy_runtime;
GRANT SELECT, INSERT, UPDATE ON hr_employee.usage_outbox TO hr_policy_runtime;
GRANT USAGE, SELECT ON SEQUENCE hr_employee.usage_outbox_outbox_id_seq TO hr_policy_runtime;
-- hr_policy_runtime: no access to chunks, releases, active_release, source_versions, sources
