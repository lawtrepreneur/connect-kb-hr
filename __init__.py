"""Shared product-neutral corpus publisher (issue #20, ADR-0005).

This package validates manifests, chunks content, builds immutable releases,
and atomically activates them in product-owned PostgreSQL databases. It is
infrastructure shared by LegalX, SignalX, and HR; it does not own
entitlements, customer data, pricing, MCP tools, or response policy, and it
contains no external DB dependency — the database is reached through the
CorpusStore protocol.
"""

from .audit import InMemoryEventLog, PublicationEvent, PublicationEventLog
from .chunker import CHUNKER_VERSION, Chunk, DeterministicChunker
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    EffectiveInterval,
    Provenance,
    validate_manifests,
)
from .publisher import (
    CorpusPublisher,
    CorpusStore,
    Embedder,
    EmergencyResult,
    PublicationResult,
)
from .release import Release, ReleaseBuilder
from .target import TargetConfig

__all__ = [
    "CHUNKER_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "Chunk",
    "CorpusManifest",
    "CorpusPublisher",
    "CorpusStore",
    "DeterministicChunker",
    "EffectiveInterval",
    "Embedder",
    "EmergencyResult",
    "InMemoryEventLog",
    "Provenance",
    "PublicationEvent",
    "PublicationEventLog",
    "PublicationResult",
    "Release",
    "ReleaseBuilder",
    "TargetConfig",
    "validate_manifests",
]
