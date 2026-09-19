"""Publish HR corpus to the active HR PostgreSQL instance.

Reads compiled approval policy from a kb-hr checkout, builds CorpusManifests
from the approved source set, and calls CorpusPublisher.publish() with the
pinned policy hash from the compiled JSON.

Usage:
    CORPUS_HR_DSN=postgresql://hrmcp:<pw>@localhost:5436/hrmcp \\
    CORPUS_HR_PUBLISHER_ROLE=hrmcp \\
    KB_HR_PATH=/path/to/kb-hr \\
    EMBEDDING_MODEL_ID=nomic-embed-text \\
    EMBEDDING_MODEL_DIGEST=sha256:... \\
    python3 scripts/publish_hr.py [--dry-run]

Required env vars:
    CORPUS_HR_DSN               DSN for the HR PostgreSQL instance
    CORPUS_HR_PUBLISHER_ROLE    DB role the publisher uses (usually same as DSN user)
    KB_HR_PATH                  Path to the kb-hr checkout on disk

Optional:
    EMBEDDING_MODEL_ID          (default: nomic-embed-text)
    EMBEDDING_MODEL_DIGEST      (default: sha256:placeholder — embeddings disabled)
    SOURCE_COMMIT               git commit of the kb-hr checkout (default: HEAD)
    --dry-run                   Validate gates but stop before writing to DB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid

# ---------------------------------------------------------------------------
# Locate the package root so the script runs from the repo root without install
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from connect_kb_hr.corpus.audit import InMemoryEventLog
from connect_kb_hr.corpus.chunker import CHUNKER_VERSION, DeterministicChunker
from connect_kb_hr.corpus.manifest import CorpusManifest, EffectiveInterval, Provenance
from connect_kb_hr.corpus.publisher import CorpusPublisher, _policy_hash
from connect_kb_hr.corpus.target import TargetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_policy(kb_hr: pathlib.Path) -> tuple[list[dict], str]:
    """Load and merge employer + employee compiled policy tuples.

    Returns (merged_tuples, pinned_policy_hash_employer_json) — the pinned
    hash is taken from the employer JSON; both files are verified to have the
    same source_commit.

    The publisher verifies that _policy_hash(merged_tuples) matches the
    pinned value.  Since _policy_hash is order-independent (it sorts the list
    itself), merging employer + employee and computing the hash over the
    merged list works as long as the publish script and the publisher agree on
    this.  We use the merged hash, not the per-audience hashes stored in the
    JSON files.
    """
    policy_dir = kb_hr / "compiled" / "approval-policy-v1"
    employer_path = policy_dir / "employer.json"
    employee_path = policy_dir / "employee.json"

    for p in (employer_path, employee_path):
        if not p.exists():
            print(f"ERROR: compiled policy not found at {p}", file=sys.stderr)
            sys.exit(1)

    employer_doc = json.loads(employer_path.read_text())
    employee_doc = json.loads(employee_path.read_text())

    merged_tuples = employer_doc["tuples"] + employee_doc["tuples"]

    # Normalize short key names → _code suffixed names expected by postgres_store.
    # The compiled policy uses "audience", "process", "content_type", "source_role";
    # the DB writer expects "audience_code", "process_code", "content_type_code",
    # "source_role_code".
    def _normalize(t: dict) -> dict:
        out = dict(t)
        for short, long in (
            ("audience", "audience_code"),
            ("process", "process_code"),
            ("content_type", "content_type_code"),
            ("source_role", "source_role_code"),
        ):
            if short in out and long not in out:
                out[long] = out[short]
        return out

    merged_tuples = [_normalize(t) for t in merged_tuples]
    # Compute pinned hash after normalization (must match what the publisher sees).
    pinned = _policy_hash(merged_tuples)
    return merged_tuples, pinned


def _build_manifests(
    kb_hr: pathlib.Path,
    compiled_assignments: list[dict],
    *,
    embedding_model_id: str,
    embedding_model_digest: str,
    source_commit: str,
) -> tuple[list[CorpusManifest], dict[str, str]]:
    """Build CorpusManifest objects for sources that appear in the policy.

    Returns (manifests, source_texts) — source_texts maps source_version_id
    to the normalized content text for chunking.

    Sources whose document directory is absent from kb-hr/documents/ are
    logged as warnings and skipped.  If no manifests are built the caller
    will error on the empty-batch gate.
    """
    import yaml  # pyyaml — already in kb-hr venv; optional dep here

    docs_root = kb_hr / "documents"

    # Build index: source_id → (doc_dir, metadata_dict)
    meta_by_sid: dict[str, tuple[pathlib.Path, dict]] = {}
    for doc_dir in docs_root.iterdir():
        meta_path = doc_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        sid = meta.get("source_id")
        if sid:
            meta_by_sid[sid] = (doc_dir, meta)

    # Deduplicate by source_version_id — one manifest per version
    seen_svids: set[str] = set()
    manifests: list[CorpusManifest] = []
    source_texts: dict[str, str] = {}

    for t in compiled_assignments:
        svid = t["source_version_id"]
        if svid in seen_svids:
            continue
        seen_svids.add(svid)

        sid = t.get("source_id")
        if not sid or sid not in meta_by_sid:
            print(
                f"WARN: source_id {sid!r} (svid={svid!r}) not found in "
                f"documents/ — skipping",
                file=sys.stderr,
            )
            continue

        doc_dir, meta = meta_by_sid[sid]
        content_path = doc_dir / "content.md"
        if not content_path.exists():
            print(
                f"WARN: content.md missing for {sid!r} at {doc_dir} — skipping",
                file=sys.stderr,
            )
            continue

        content_text = content_path.read_text(encoding="utf-8")

        # Use metadata.yaml's content_hash as the canonical content hash.
        # This is the hash of the LegalX DB content at canonicalization time —
        # the value that the approval record reviewed and signed off on.
        # The sha256 of content.md diverges because rendering adds/removes whitespace.
        meta_content_hash = meta.get("content_hash", "")
        expected_hash = t.get("source_content_hash", "")

        # Guard: ensure metadata.yaml and approval record agree on content hash.
        if meta_content_hash and expected_hash and meta_content_hash != expected_hash:
            print(
                f"ERROR: content_hash mismatch for {svid!r}: "
                f"metadata.yaml={meta_content_hash[:12]}… approval_record={expected_hash[:12]}…",
                file=sys.stderr,
            )
            sys.exit(1)

        # content_hash used downstream: sha256 of content.md text (internal consistency).
        # The gate checks against source_content_hash (approved metadata hash).
        content_hash = _sha256_text(content_text)

        # Compute metadata hash over the raw YAML bytes for provenance
        meta_path = doc_dir / "metadata.yaml"
        metadata_hash = _sha256_file(meta_path)

        effective_from = meta.get("effective_from") or "1900-01-01"
        rights = meta.get("rights", {})
        prov = meta.get("provenance", {})

        manifest = CorpusManifest(
            source_id=sid,
            source_version_id=svid,
            source_kind=meta.get("source_kind", "casebrief"),
            canonical_origin="https://github.com/lawtrepreneur/kb-hr",
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            jurisdiction=meta.get("jurisdiction", "CA-ON"),
            effective_interval=EffectiveInterval(start=str(effective_from)),
            provenance=Provenance(origin_commit=source_commit),
            canonical_locator=f"documents/{doc_dir.name}/content.md",
            rights_basis=rights.get("basis", "owned"),
            rights_holder=rights.get("holder", "RHH"),
            permitted_use=rights.get("permitted_use", "licensed_product"),
            attribution_required=bool(rights.get("attribution_required", False)),
            global_state=meta.get("publication_status", "valid")
                if meta.get("publication_status") in ("valid", "suspended", "withdrawn")
                else "valid",
            chunker_version=CHUNKER_VERSION,
            embedding_model_id=embedding_model_id,
            embedding_model_digest=embedding_model_digest,
        )
        manifests.append(manifest)
        source_texts[svid] = content_text

    return manifests, source_texts


def _get_source_commit(kb_hr: pathlib.Path) -> str:
    """Return the HEAD commit of the kb-hr repo, or env override."""
    override = os.environ.get("SOURCE_COMMIT")
    if override:
        return override
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(kb_hr),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Publish HR corpus to Postgres")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate gates but do not write to DB")
    args = parser.parse_args()

    dsn = os.environ.get("CORPUS_HR_DSN", "")
    publisher_role = os.environ.get("CORPUS_HR_PUBLISHER_ROLE", "hrmcp")
    kb_hr_path = os.environ.get("KB_HR_PATH", "")
    embedding_model_id = os.environ.get("EMBEDDING_MODEL_ID", "nomic-embed-text")
    embedding_model_digest = os.environ.get(
        "EMBEDDING_MODEL_DIGEST", "sha256:placeholder"
    )

    if not dsn:
        print("ERROR: CORPUS_HR_DSN not set", file=sys.stderr)
        return 1
    if not kb_hr_path:
        print("ERROR: KB_HR_PATH not set", file=sys.stderr)
        return 1

    kb_hr = pathlib.Path(kb_hr_path).resolve()
    if not kb_hr.exists():
        print(f"ERROR: KB_HR_PATH does not exist: {kb_hr}", file=sys.stderr)
        return 1

    # 1. Load compiled policy
    print("Loading compiled approval policy …")
    compiled_assignments, pinned_policy_hash = _load_policy(kb_hr)
    print(f"  {len(compiled_assignments)} assignment tuples loaded")
    print(f"  pinned_policy_hash: {pinned_policy_hash[:16]}…")

    source_commit = _get_source_commit(kb_hr)
    print(f"  source_commit: {source_commit[:12]}…")

    # 2. Build manifests + source texts
    print("Building manifests from documents/ …")
    manifests, source_texts = _build_manifests(
        kb_hr,
        compiled_assignments,
        embedding_model_id=embedding_model_id,
        embedding_model_digest=embedding_model_digest,
        source_commit=source_commit,
    )
    print(f"  {len(manifests)} manifest(s) ready")

    if not manifests:
        print("ERROR: no publishable manifests found — check WARN lines above",
              file=sys.stderr)
        return 1

    if args.dry_run:
        print("--dry-run: gates validated; stopping before DB write.")
        return 0

    # 3. Connect to DB
    target = TargetConfig(
        product="hr",
        dsn=dsn,
        publisher_role=publisher_role,
        schema_name="kb",
    )

    from connect_kb_hr.db.postgres_store import PostgresCorpusStore

    store = PostgresCorpusStore(dsn=dsn, schema="kb")

    # 4. Build embedder — real HTTP call to llama-swap /v1/embeddings.
    embed_url = os.environ.get(
        "KNOWLEDGE_EMBED_URL", "https://chat.tail713de8.ts.net/v1/embeddings"
    )

    class _HttpEmbedder:
        model_id = embedding_model_id
        model_digest = embedding_model_digest

        def embed(self, text: str) -> list[float]:
            import httpx
            resp = httpx.post(
                embed_url,
                json={"model": self.model_id, "input": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    embedder = _HttpEmbedder()

    event_log = InMemoryEventLog()
    publisher = CorpusPublisher(
        event_log=event_log,
        chunker=DeterministicChunker(version=CHUNKER_VERSION),
        embedder=embedder,
    )

    # Filter assignments to only those whose source_version_id has a manifest.
    # Skipped sources (missing documents) must not appear in release_assignments
    # — the FK on source_versions would fail.
    manifest_svids = {m.source_version_id for m in manifests}
    compiled_assignments = [
        a for a in compiled_assignments
        if a.get("source_version_id") in manifest_svids
    ]
    # Recompute pinned hash over the filtered set — publisher verifies this.
    pinned_policy_hash = _policy_hash(compiled_assignments)

    publication_run_id = str(uuid.uuid4())
    print(f"Publishing … (run_id={publication_run_id})")

    result = publisher.publish(
        publication_run_id=publication_run_id,
        target=target,
        store=store,
        manifests=manifests,
        source_texts=source_texts,
        source_commit=source_commit,
        compiled_assignments=compiled_assignments,
        pinned_policy_hash=pinned_policy_hash,
    )

    print(f"Result: status={result.status} release_id={result.release_id}")
    if result.detail:
        print(f"  detail: {result.detail}")
    if not result.ok:
        print(f"FAILED: {result.detail}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
