# Changelog

## [Unreleased]

## [0.2.0] — 2026-09-19

### Added
- `DeterministicChunker` v1.1: `_split_long()` sub-splits sections exceeding 500 chars at sentence boundaries (empirically measured llama-swap endpoint limit is ~508 chars)
- Real HTTP embedder in `scripts/publish_hr.py` — calls llama-swap `/v1/embeddings` via `httpx`; replaces zero-vector stub
- `CHUNKER_VERSION` constant imported and used throughout `publish_hr.py` (was hardcoded `"1.0"`)

### Fixed
- `publish_hr.py`: normalize compiled policy tuple keys (`audience` → `audience_code` etc.) before passing to `postgres_store`
- `publish_hr.py`: filter `compiled_assignments` to manifested `source_version_id`s only — prevents FK violation when hiring sources are missing from `documents/`
- `publish_hr.py`: recompute `pinned_policy_hash` after filtering (was computed pre-filter, causing `policy_hash_mismatch`)
- `publisher.py`: content-hash gate checks `source_content_hash` (approved hash) with fallback to `content_hash`
- `postgres_store.py`: install the updated unified-schema version in `mcp-hr` venv — old installed package used split `hr_employer`/`hr_employee` schema, causing `ValueError` at retrieval time

### Notes
- On `CHUNKER_VERSION` bump, existing corpus tables must be cleared before re-publish to avoid `UniqueViolation` on `(source_version_id, ordinal)` unique constraint
