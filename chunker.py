"""Deterministic chunking for immutable source versions (issue #20, ADR-0005 section 7).

Chunk boundaries are source-kind-specific:

- ``clause`` — the whole clause stays a single chunk;
- ``casebrief`` — split on structured section markers (``## `` headings);
- ``rhh_document`` — split on heading-bounded sections (``# ``/``## `` headings);
- ``legislation`` — split on provision boundaries (``Section N`` lines).

The chunker is pure and deterministic: the same text, source kind, and
chunker version always produce the same chunk IDs and hashes, so releases
built from the same manifest are byte-identical.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

CHUNKER_VERSION = "1.0"

_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,2}\s+(.*)$", re.MULTILINE)
_PROVISION_RE = re.compile(r"^(?:Section\s+\d+|Part\s+\d+|Schedule\s+\d+)[^\n]*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One immutable chunk derived from an immutable source version.

    ``chunk_id`` is derived from the source version, ordinal, and chunker
    version, so it is stable across rebuilds. ``content`` is the chunk text
    and never leaves the build process into logs or audit events.
    """

    chunk_id: str
    source_version_id: str
    source_locator: str
    ordinal: int
    content: str
    content_hash: str
    chunker_version: str


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_id(source_version_id: str, ordinal: int, content_hash: str, chunker_version: str) -> str:
    basis = f"{source_version_id}|{ordinal}|{content_hash}|{chunker_version}"
    return _sha256_hex(basis.encode("utf-8"))


class DeterministicChunker:
    """Splits source text into immutable, source-kind-specific chunks."""

    def __init__(self, version: str = CHUNKER_VERSION) -> None:
        self.version = version

    def chunk(self, source_version_id: str, source_kind: str, text: str, locator_prefix: str = "") -> list[Chunk]:
        """Split ``text`` into chunks for ``source_kind``.

        ``locator_prefix`` is an optional stable path prefix (e.g. a file
        path within the canonical origin) recorded on every chunk.
        """
        segments = self._split(source_kind, text)
        if not segments:
            segments = [""]
        chunks: list[Chunk] = []
        for ordinal, segment in enumerate(segments):
            content = segment.strip()
            if not content:
                continue
            content_hash = _sha256_hex(content.encode("utf-8"))
            locator = f"{locator_prefix}#{ordinal}" if locator_prefix else f"#{ordinal}"
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_version_id, ordinal, content_hash, self.version),
                    source_version_id=source_version_id,
                    source_locator=locator,
                    ordinal=ordinal,
                    content=content,
                    content_hash=content_hash,
                    chunker_version=self.version,
                )
            )
        return chunks

    def _split(self, source_kind: str, text: str) -> list[str]:
        if source_kind == "clause":
            # Clauses stay whole; a clause is exactly one chunk.
            return [text]
        if source_kind == "casebrief":
            return self._split_on(text, _SECTION_RE)
        if source_kind == "rhh_document":
            return self._split_on(text, _HEADING_RE)
        if source_kind == "legislation":
            return self._split_on(text, _PROVISION_RE)
        raise ValueError(f"unsupported source_kind for chunking: {source_kind!r}")

    @staticmethod
    def _split_on(text: str, pattern: re.Pattern[str]) -> list[str]:
        """Split on section boundaries, keeping the boundary with its body."""
        matches = list(pattern.finditer(text))
        if not matches:
            return [text]
        segments: list[str] = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                segments.append(preamble)
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segments.append(text[start:end])
        return segments

    @staticmethod
    def verify_chunks(chunks: Sequence[Chunk]) -> list[str]:
        """Re-hash chunk contents and check ordinal/chunk_id consistency.

        Returns a list of error strings (empty when all chunks verify).
        """
        errors: list[str] = []
        seen_ordinal: set[int] = set()
        for c in chunks:
            expected = _sha256_hex(c.content.encode("utf-8"))
            if expected != c.content_hash:
                errors.append(f"chunk {c.chunk_id}: content_hash mismatch")
            expected_id = _chunk_id(c.source_version_id, c.ordinal, c.content_hash, c.chunker_version)
            if expected_id != c.chunk_id:
                errors.append(f"chunk {c.chunk_id}: chunk_id mismatch")
            if c.ordinal in seen_ordinal:
                errors.append(f"chunk {c.chunk_id}: duplicate ordinal {c.ordinal}")
            seen_ordinal.add(c.ordinal)
        return errors
