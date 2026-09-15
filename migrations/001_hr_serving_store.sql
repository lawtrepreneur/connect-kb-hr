-- Migration 001: Unified corpus/policy design (ADR-0005 superseded)
-- connect-kb-hr  — replaces audience-split hr_employer/hr_employee schema
--
-- Design decisions (all locked):
--   - ONE schema: kb  (shared corpus tables; no audience split)
--   - ONE policy schema: hr_policy  (entitlement + content-free usage)
--   - ONE shared chunks table + ONE HNSW index
--   - Audience = assignment predicate in kb.release_assignments, never a schema/DSN/role selector
--
-- Roles created (idempotent):
--   hr_migrator      — DDL only
--   hr_publisher     — write kb schema
--   hr_runtime       — EXECUTE on api functions only; SELECT on kb corpus tables; no direct table writes
--   hr_policy_writer — write hr_policy schema
--
-- Schema version tag: 1.0
-- Prerequisites: PostgreSQL >= 14 with pgvector extension available
-- Rollback: see 001_rollback.sql

-- ---------------------------------------------------------------------------
-- 0. Roles (idempotent)
-- ---------------------------------------------------------------------------

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_migrator') THEN
    CREATE ROLE hr_migrator NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_publisher') THEN
    CREATE ROLE hr_publisher NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_runtime') THEN
    CREATE ROLE hr_runtime NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_policy_writer') THEN
    CREATE ROLE hr_policy_writer NOLOGIN;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 1. Enable pgvector (idempotent)
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 2. Schema: kb  (unified corpus)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS kb;

-- ---------------------------------------------------------------------------
-- 2a. Migration tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.schema_version (
    version         TEXT        NOT NULL,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    description     TEXT        NOT NULL
);

INSERT INTO kb.schema_version (version, description)
VALUES ('1.0', 'unified corpus/policy design: single kb schema, audience as assignment predicate')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2b. Controlled vocabulary tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.audience (
    audience_code TEXT PRIMARY KEY
);

INSERT INTO kb.audience (audience_code)
VALUES ('employer'), ('employee')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS kb.process (
    process_code TEXT PRIMARY KEY
);

INSERT INTO kb.process (process_code)
VALUES
    ('hiring'),
    ('onboarding'),
    ('performance'),
    ('leave'),
    ('discipline'),
    ('termination')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS kb.content_type (
    content_type_code TEXT PRIMARY KEY
);

INSERT INTO kb.content_type (content_type_code)
VALUES
    ('question'),
    ('checklist'),
    ('plan'),
    ('template')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS kb.source_role (
    source_role_code TEXT PRIMARY KEY
);

INSERT INTO kb.source_role (source_role_code)
VALUES
    ('supporting_authority'),
    ('substantive_guidance'),
    ('reusable_artifact')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2c. Sources: stable source identity (content-free metadata)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.sources (
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

-- ---------------------------------------------------------------------------
-- 2d. Source versions: immutable
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.source_versions (
    source_version_id       TEXT        NOT NULL,
    source_id               TEXT        NOT NULL REFERENCES kb.sources(source_id),
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
    CONSTRAINT source_versions_pk PRIMARY KEY (source_version_id),
    CONSTRAINT source_versions_unique_content UNIQUE (source_id, content_hash)
);

CREATE INDEX IF NOT EXISTS source_versions_source_id_idx
    ON kb.source_versions (source_id);
CREATE INDEX IF NOT EXISTS source_versions_global_state_idx
    ON kb.source_versions (global_state);

-- ---------------------------------------------------------------------------
-- 2e. Chunks: one row per unique (chunk_content, embedding_model); no audience duplication
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.chunks (
    chunk_id            TEXT        NOT NULL,
    source_version_id   TEXT        NOT NULL REFERENCES kb.source_versions(source_version_id),
    source_locator      TEXT        NOT NULL,
    ordinal             INTEGER     NOT NULL,
    content_hash        TEXT        NOT NULL,
    chunker_version     TEXT        NOT NULL,
    embedding           vector(768),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_pk PRIMARY KEY (chunk_id),
    CONSTRAINT chunks_unique_ordinal UNIQUE (source_version_id, ordinal)
);

-- HNSW index for semantic search (pgvector cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON kb.chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- 2f. Releases: one release covers the full corpus (no audience split)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.releases (
    release_id              TEXT        NOT NULL,
    publication_run_id      TEXT        NOT NULL,
    manifest_hash           TEXT        NOT NULL,
    policy_hash             TEXT        NOT NULL,
    source_commit           TEXT        NOT NULL,
    chunker_version         TEXT        NOT NULL,
    embedding_model_id      TEXT        NOT NULL,
    build_status            TEXT        NOT NULL CHECK (build_status IN ('building','built','failed')),
    validation_status       TEXT        NOT NULL CHECK (validation_status IN ('pending','passed','failed')),
    chunk_count             INTEGER     NOT NULL DEFAULT 0,
    assignment_count        INTEGER     NOT NULL DEFAULT 0,
    activated_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT releases_pk PRIMARY KEY (release_id)
);

CREATE INDEX IF NOT EXISTS releases_validation_status_idx
    ON kb.releases (validation_status, created_at DESC);

-- ---------------------------------------------------------------------------
-- 2g. Active-release pointer: singleton — exactly one row
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.active_release (
    singleton               BOOLEAN     NOT NULL DEFAULT TRUE CHECK (singleton),
    release_id              TEXT        NOT NULL REFERENCES kb.releases(release_id),
    activated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT active_release_singleton PRIMARY KEY (singleton)
);

-- ---------------------------------------------------------------------------
-- 2h. Release assignments: compiled policy tuples per release
--     No wildcards, no arrays; audience is a data column not a schema selector
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kb.release_assignments (
    release_id              TEXT        NOT NULL REFERENCES kb.releases(release_id),
    source_version_id       TEXT        NOT NULL REFERENCES kb.source_versions(source_version_id),
    audience_code           TEXT        NOT NULL REFERENCES kb.audience(audience_code),
    process_code            TEXT        NOT NULL REFERENCES kb.process(process_code),
    content_type_code       TEXT        NOT NULL REFERENCES kb.content_type(content_type_code),
    source_role_code        TEXT        NOT NULL REFERENCES kb.source_role(source_role_code),
    approval_record         TEXT        NOT NULL,
    reviewer_id             TEXT        NOT NULL,
    reviewed_at             TEXT        NOT NULL,
    PRIMARY KEY (release_id, source_version_id, audience_code, process_code, content_type_code, source_role_code)
);

-- ---------------------------------------------------------------------------
-- 3. Schema: hr_policy  (entitlement + content-free usage tracking)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS hr_policy;

-- ---------------------------------------------------------------------------
-- 3a. Migration tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hr_policy.schema_version (
    version         TEXT        NOT NULL,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    description     TEXT        NOT NULL
);

INSERT INTO hr_policy.schema_version (version, description)
VALUES ('1.0', 'unified corpus/policy design: content-free usage tracking, audience as data column')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3b. Usage sequences
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hr_policy.usage_sequences (
    usage_ref           TEXT        NOT NULL,
    customer_id         TEXT        NOT NULL,
    audience_code       TEXT        NOT NULL REFERENCES kb.audience(audience_code),
    process_code        TEXT        NOT NULL REFERENCES kb.process(process_code),
    content_type_code   TEXT        NOT NULL REFERENCES kb.content_type(content_type_code),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ NOT NULL,
    CONSTRAINT usage_sequences_pk PRIMARY KEY (usage_ref)
);

-- ---------------------------------------------------------------------------
-- 3c. Usage events
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hr_policy.usage_events (
    usage_event_id      TEXT        NOT NULL,
    usage_ref           TEXT        NOT NULL REFERENCES hr_policy.usage_sequences(usage_ref),
    customer_id         TEXT        NOT NULL,
    idempotency_key     TEXT        NOT NULL,
    event_type          TEXT        NOT NULL CHECK (event_type IN ('initial','clarification','refinement','completed')),
    process_code        TEXT        NOT NULL REFERENCES kb.process(process_code),
    content_type_code   TEXT        NOT NULL REFERENCES kb.content_type(content_type_code),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT usage_events_pk PRIMARY KEY (usage_event_id),
    CONSTRAINT usage_events_idempotency UNIQUE (customer_id, idempotency_key)
);

-- ---------------------------------------------------------------------------
-- 3d. Usage outbox (reliable delivery)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hr_policy.usage_outbox (
    outbox_id           BIGSERIAL   PRIMARY KEY,
    usage_event_id      TEXT        NOT NULL REFERENCES hr_policy.usage_events(usage_event_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at        TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- 4. Grants
-- ---------------------------------------------------------------------------

-- hr_publisher: full write on kb schema
GRANT USAGE ON SCHEMA kb TO hr_publisher;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA kb TO hr_publisher;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA kb TO hr_publisher;

-- hr_runtime: USAGE on kb schema + SELECT on corpus tables (no table writes)
GRANT USAGE ON SCHEMA kb TO hr_runtime;
GRANT SELECT ON kb.chunks             TO hr_runtime;
GRANT SELECT ON kb.source_versions    TO hr_runtime;
GRANT SELECT ON kb.sources            TO hr_runtime;
GRANT SELECT ON kb.releases           TO hr_runtime;
GRANT SELECT ON kb.active_release     TO hr_runtime;
GRANT SELECT ON kb.release_assignments TO hr_runtime;
GRANT SELECT ON kb.audience           TO hr_runtime;
GRANT SELECT ON kb.process            TO hr_runtime;
GRANT SELECT ON kb.content_type       TO hr_runtime;
GRANT SELECT ON kb.source_role        TO hr_runtime;
GRANT SELECT ON kb.schema_version     TO hr_runtime;

-- hr_policy_writer: full write on hr_policy schema
GRANT USAGE ON SCHEMA hr_policy TO hr_policy_writer;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA hr_policy TO hr_policy_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA hr_policy TO hr_policy_writer;
