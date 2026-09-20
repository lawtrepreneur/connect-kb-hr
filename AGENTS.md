# Agent Instructions — connect-kb-hr

## Wiki / docs

All documentation for this repo is symlinked into the second brain at
`~/workspace/secondbrain/internal/connect-kb-hr/docs/`.

| Doc | Wiki path |
|---|---|
| Technical specification (ADR-001) | `~/workspace/secondbrain/internal/connect-kb-hr/docs/adr/0001-techspec.md` |
| Changelog | `~/workspace/secondbrain/internal/connect-kb-hr/docs/CHANGELOG.md` |
| Ecosystem overview (full pipeline) | `~/workspace/secondbrain/internal/kb-hr/docs/ecosystem-overview.md` |

---

## Purpose

`connect-kb-hr` is the HR corpus publication pipeline. It reads canonical sources
from `kb-hr`, chunks and embeds them, and publishes atomically to HR PostgreSQL/pgvector
for `mcp-hr` to serve.

It does **not** own the canonical sources (`kb-hr`), the MCP server (`mcp-hr`), or
the database schema contract (`db-hr`).

---

## Repo layout

```
src/connect_kb_hr/
├── corpus/
│   ├── publisher.py   — CorpusPublisher: validate, chunk, embed, activate
│   ├── manifest.py    — CorpusManifest: versioned source-version contract
│   ├── chunker.py     — DeterministicChunker v1.1 (500-char limit, sentence-boundary splits)
│   ├── release.py     — Release / ReleaseBuilder: immutable release assembly
│   ├── target.py      — TargetConfig: product-owned DB binding
│   └── audit.py       — PublicationEventLog: content-free audit trail
├── db/
│   ├── postgres_store.py  — CorpusStore protocol impl (unified kb schema)
│   ├── memory_store.py    — In-memory store for tests
│   └── migrate.py         — Schema migration runner
└── migrations/
    └── 001_hr_serving_store.sql  — Unified kb schema + roles
scripts/
└── publish_hr.py      — End-to-end publish entrypoint (reads kb-hr, publishes to PG)
```

---

## Key env vars

| Var | Purpose |
|---|---|
| `CORPUS_HR_DSN` | HR PostgreSQL DSN (also accepted as `HR_MCP_PG_URL`) |
| `CORPUS_HR_PUBLISHER_ROLE` | DB role for publish operations (usually `hrmcp`) |
| `KB_HR_PATH` | Absolute path to `kb-hr` checkout |
| `KNOWLEDGE_EMBED_URL` | Embedding endpoint (default: Tailscale llama-swap) |
| `EMBEDDING_MODEL_ID` | Embedding model (default: `nomic-embed-text`) |
| `EMBEDDING_MODEL_DIGEST` | Model digest for release record |

---

## Publish command

```bash
cd /mnt/docker-ssd/romeshh/sites/connect-kb-hr
set -a && source /mnt/docker-ssd/romeshh/sites/mcp-hr/.env && set +a
KB_HR_PATH=/mnt/docker-ssd/romeshh/sites/kb-hr \
CORPUS_HR_PUBLISHER_ROLE=hrmcp \
EMBEDDING_MODEL_ID=nomic-embed-text \
EMBEDDING_MODEL_DIGEST=sha256:nomic-embed-text-v1.5 \
.venv/bin/python scripts/publish_hr.py
```

**Note:** On `CHUNKER_VERSION` bump, clear `kb.chunks`, `kb.source_versions`,
`kb.sources`, `kb.release_assignments`, `kb.releases`, `kb.active_release` before
re-publishing to avoid `UniqueViolation` on `(source_version_id, ordinal)`.

---

## Related repos

| Repo | Path | Role |
|---|---|---|
| `kb-hr` | `/mnt/docker-ssd/romeshh/sites/kb-hr` | Canonical editorial sources |
| `mcp-hr` | `/mnt/docker-ssd/romeshh/sites/mcp-hr` | MCP serving layer |
| `db-hr` | `/mnt/docker-ssd/romeshh/sites/db-hr` | DB contract/migrations |
