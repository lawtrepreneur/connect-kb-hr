# connect-kb-hr

HR knowledge publication pipeline: compiles `kb-hr` canonical sources into PostgreSQL/pgvector releases for `mcp-hr`.

**Repository boundary (ADR-0005):**
- **Owns:** HR corpus publication, PostgreSQL migrations, embedder configuration, release activation
- **Reads from:** `kb-hr` canonical sources and assignments
- **Writes to:** HR PostgreSQL database (separate from `launch-mcp`)
- **Does not own:** MCP server tooling (that's `mcp-hr`), generic authentication (that's `launch-mcp`)

## Architecture

```
kb-hr (canonical sources)
    ↓
connect-kb-hr (this repo)
    ├─ read kb-hr documents and assignments
    ├─ compile manifests
    ├─ chunk and embed
    └─ publish to HR PostgreSQL/pgvector
        ├─ hr_employer schema
        └─ hr_employee schema
    ↓
mcp-hr (read-only query layer)
    └─ serves entitled consultants
```

## Status

**Bootstrap phase:** Preparing to move `launch_mcp.corpus` module while preserving history.
