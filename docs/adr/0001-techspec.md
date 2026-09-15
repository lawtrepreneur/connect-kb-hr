# ADR-001: Techspec

## Status

Accepted

## Date

2026-09-15

## Context

connect-kb-hr is the HR knowledge publication pipeline. It reads canonical sources from `kb-hr`, compiles them into immutable releases (manifests, chunks, embeddings), and publishes atomically to a product-owned PostgreSQL/pgvector database for `mcp-hr` to serve.

This repo is product-neutral in design (ADR-0005 unified corpus): the same `CorpusPublisher` core serves LegalX, SignalX, and HR — each product supplies its own `TargetConfig` with separate credentials and schema.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Database | PostgreSQL 14+ with pgvector |
| DB driver | psycopg2-binary |
| Build system | hatchling |
| Validation | jsonschema (manifest validation) |
| Testing | pytest |
| Linting | ruff (E, F, I, UP rules) |
| Package manager | uv |

## Architecture

```
kb-hr (canonical Git sources)
    ↓
connect-kb-hr (this repo — publication pipeline)
    ├─ corpus/
    │   ├─ publisher.py    ← CorpusPublisher: validate, chunk, embed, activate
    │   ├─ manifest.py     ← CorpusManifest: versioned source-version contract
    │   ├─ chunker.py      ← DeterministicChunker: source-kind-specific chunking
    │   ├─ release.py      ← Release / ReleaseBuilder: immutable release assembly
    │   ├─ target.py       ← TargetConfig: product-owned DB binding (env-only creds)
    │   └─ audit.py        ← PublicationEventLog: content-free audit trail
    ├─ db/
    │   ├─ postgres_store.py  ← CorpusStore protocol impl (PostgreSQL)
    │   ├─ memory_store.py    ← In-memory store for tests
    │   └─ migrate.py         ← Schema migration runner
    └─ migrations/
        └─ 001_hr_serving_store.sql  ← Unified kb schema + roles
    ↓
mcp-hr (read-only query layer — separate repo)
```

## Data model (PostgreSQL)

**Schema `kb`** (unified corpus — no audience split):
- `kb.schema_version` — migration tracking
- `kb.audience` — controlled vocabulary (employer, employee)
- `kb.process` — controlled vocabulary (hiring, onboarding, performance, leave, discipline, termination)
- `kb.source_versions` — immutable source version records (manifest fields)
- `kb.chunks` — deterministic chunks with HNSW vector index
- `kb.releases` — immutable release records (manifest_hash, policy_hash, status)
- `kb.release_assignments` — audience/process/content-type predicates per release

**Schema `hr_policy`** (entitlement + usage):
- Entitlement records linking users/consultants to entitled content

**Roles** (least-privilege):
- `hr_migrator` — DDL only
- `hr_publisher` — write `kb` schema
- `hr_runtime` — EXECUTE on API functions; SELECT on corpus tables; no direct writes
- `hr_policy_writer` — write `hr_policy` schema

## Key choices

1. **Product-neutral publisher core** — one `CorpusPublisher` class serves all products; product differentiation is via `TargetConfig` (DSN, role, schema name). No product-specific code in the core.
2. **Audience as assignment predicate, not schema split** — ADR-0005 supersedes the earlier `hr_employer`/`hr_employee` split. One `kb` schema; audience is a row in `kb.release_assignments`.
3. **Immutable releases** — a Release is built once, validated, and activated atomically. No in-place mutation. Rollback = activate prior release.
4. **Deterministic chunking** — same source text + source kind + chunker version always produces identical chunk IDs and hashes. Enables byte-identical rebuilds.
5. **Content-free manifests** — manifests carry hashes and metadata, never source body text. The publisher verifies source body matches manifest content hash before building.
6. **Protocol-based store abstraction** — `CorpusStore` protocol decouples the publisher from PostgreSQL; tests use `MemoryStore`, production uses `PostgresStore`.
7. **Environment-only credentials** — `TargetConfig.from_env()` reads `CORPUS_<PRODUCT>_DSN` and `CORPUS_<PRODUCT>_PUBLISHER_ROLE`. No config files, no logged DSNs.
8. **Content-free audit events** — `PublicationEventLog` records identifiers and reason codes only; never source bodies, DSNs, or credentials.

## Repository boundary (ADR-0005)

| Owns | Does not own |
|---|---|
| HR corpus publication pipeline | MCP server tooling (`mcp-hr`) |
| PostgreSQL migrations for HR corpus | Generic authentication (`launch-mcp`) |
| Embedder configuration | Canonical source content (`kb-hr`) |
| Release activation/rollback | Serving/query logic (`mcp-hr`) |

## References

- `migrations/001_hr_serving_store.sql` — full schema DDL
- `src/connect_kb_hr/corpus/publisher.py` — publisher core
- `src/connect_kb_hr/corpus/target.py` — product target binding
- `src/connect_kb_hr/corpus/manifest.py` — manifest contract
- `src/connect_kb_hr/corpus/chunker.py` — deterministic chunking
- `src/connect_kb_hr/corpus/release.py` — release assembly
- `kb-hr/schema/corpus-manifest-v1.0.json` — upstream manifest schema
- `kb-hr/TAXONOMY.md` — controlled vocabulary
