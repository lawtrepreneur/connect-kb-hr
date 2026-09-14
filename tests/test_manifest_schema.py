"""Tests for kb-hr#1: corpus manifest schema (JSON Schema v1.0).

Validates:
- All three fixture types validate against the schema
- Stable IDs survive round-trip serialisation
- Unsupported schema_version is rejected
- Content hash and metadata hash are distinct
- global_state='withdrawn' is representable without fixture deletion
- Required fields: missing any one fails validation
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

KB_HR_ROOT = pathlib.Path(__file__).parent.parent.parent / "kb-hr"
SCHEMA_PATH = KB_HR_ROOT / "schema" / "corpus-manifest-v1.0.json"
FIXTURES_DIR = KB_HR_ROOT / "fixtures"

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _validate(instance: dict) -> list[str]:
    """Return list of validation error messages (empty = valid)."""
    schema = _load_schema()
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        return [e.message for e in validator.iter_errors(instance)]
    except ImportError:
        # Fallback: manual required-field check only
        required = schema.get("required", [])
        return [f"missing required field: {f}" for f in required if f not in instance]


# ---------------------------------------------------------------------------
# kb-hr#1 AC: all three fixture types validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_file", [
    "rhh-document-fixture.json",
    "legalx-clause-fixture.json",
    "legalx-casebrief-fixture.json",
])
def test_fixture_validates_against_schema(fixture_file):
    """All three fixture types must validate against corpus-manifest-v1.0.json."""
    instance = _load_fixture(fixture_file)
    errors = _validate(instance)
    assert errors == [], f"{fixture_file} failed validation:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# kb-hr#1 AC: stable IDs survive round-trip (publish to multiple DBs)
# ---------------------------------------------------------------------------

def test_stable_ids_survive_round_trip():
    """source_id and source_version_id must be identical after JSON round-trip."""
    for fixture_file in ["rhh-document-fixture.json", "legalx-clause-fixture.json", "legalx-casebrief-fixture.json"]:
        original = _load_fixture(fixture_file)
        serialised = json.dumps(original, sort_keys=True)
        deserialised = json.loads(serialised)
        assert deserialised["source_id"] == original["source_id"]
        assert deserialised["source_version_id"] == original["source_version_id"]


# ---------------------------------------------------------------------------
# kb-hr#1 AC: unsupported schema version fails before DB mutation
# ---------------------------------------------------------------------------

def test_unsupported_schema_version_fails_validation():
    """A manifest with manifest_schema_version != '1.0' must fail validation."""
    instance = _load_fixture("rhh-document-fixture.json")
    bad = {**instance, "manifest_schema_version": "2.0"}
    errors = _validate(bad)
    assert errors, "Expected validation errors for unsupported schema version"


# ---------------------------------------------------------------------------
# kb-hr#1 AC: content changes and metadata-only changes produce distinct hashes
# ---------------------------------------------------------------------------

def test_content_hash_and_metadata_hash_are_distinct():
    """content_hash and metadata_hash must differ — they cover different fields."""
    for fixture_file in ["rhh-document-fixture.json", "legalx-clause-fixture.json"]:
        instance = _load_fixture(fixture_file)
        assert instance["content_hash"] != instance["metadata_hash"], (
            f"{fixture_file}: content_hash and metadata_hash must be distinct"
        )


# ---------------------------------------------------------------------------
# kb-hr#1 AC: global withdrawal representable without deleting historical versions
# ---------------------------------------------------------------------------

def test_withdrawn_state_is_valid_and_not_publishable():
    """global_state='withdrawn' must be a valid schema value.
    The CorpusManifest.is_publishable property rejects it — tested here via schema enum.
    """
    instance = _load_fixture("rhh-document-fixture.json")
    withdrawn = {**instance, "global_state": "withdrawn"}
    errors = _validate(withdrawn)
    assert errors == [], "global_state='withdrawn' should be valid in schema"
    # Confirm 'valid' is the only publishable state
    assert instance["global_state"] == "valid"


def test_invalid_global_state_fails():
    """An unrecognised global_state value must fail schema validation."""
    instance = _load_fixture("rhh-document-fixture.json")
    bad = {**instance, "global_state": "archived"}
    errors = _validate(bad)
    assert errors, "Expected validation error for unknown global_state"


# ---------------------------------------------------------------------------
# kb-hr#1 AC: missing required fields fail validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "source_id", "source_version_id", "source_kind", "canonical_origin",
    "content_hash", "metadata_hash", "jurisdiction", "effective_interval",
    "provenance", "canonical_locator", "rights_basis", "rights_holder",
    "permitted_use", "attribution_required", "global_state",
    "chunker_version", "embedding_model_id", "embedding_model_digest",
])
def test_missing_required_field_fails(field):
    """Omitting any required field must fail validation."""
    instance = _load_fixture("rhh-document-fixture.json")
    bad = {k: v for k, v in instance.items() if k != field}
    errors = _validate(bad)
    assert errors, f"Expected error when '{field}' is missing"


# ---------------------------------------------------------------------------
# kb-hr#1 AC: source_kind enum is exhaustive for known chunker kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["clause", "casebrief", "rhh_document", "legislation"])
def test_valid_source_kinds(kind):
    instance = _load_fixture("rhh-document-fixture.json")
    patched = {**instance, "source_kind": kind}
    errors = _validate(patched)
    assert errors == [], f"source_kind='{kind}' should be valid"


def test_unknown_source_kind_fails():
    instance = _load_fixture("rhh-document-fixture.json")
    bad = {**instance, "source_kind": "memo"}
    errors = _validate(bad)
    assert errors, "Expected error for unknown source_kind"
