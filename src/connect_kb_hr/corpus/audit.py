"""Append-only publication event log (issue #20, ADR-0005 sections 5, 11).

Publication events record what the publisher actually did: built, validated,
activated, suspended, withdrew, or rolled back. The log is append-only and
content-free: events carry identifiers, hashes, and controlled status strings
only — never source bodies, DSNs, credentials, or request/response payloads.

The in-memory implementation is the launch form; a DB-backed store implements
the same PublicationEventLog protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

EventKind = Literal[
    "build_started",
    "build_completed",
    "build_failed",
    "validation_passed",
    "validation_failed",
    "release_activated",
    "release_rolled_back",
    "source_suspended",
    "source_withdrawn",
    "idempotent_noop",
    "target_rejected",
]


@dataclass(frozen=True)
class PublicationEvent:
    """One append-only publication event.

    All fields are safe to persist and log: no source text, no connection
    strings, no credentials. ``detail`` is a controlled short reason string
    (e.g. a validation error category), never a raw error body.
    """

    event_id: str
    publication_run_id: str
    product: str
    audience: str
    kind: EventKind
    release_id: str | None = None
    manifest_hash: str | None = None
    detail: str | None = None
    timestamp: str = ""


class PublicationEventLog(Protocol):
    """Consumer-supplied append-only event log."""

    def append(self, event: PublicationEvent) -> None:
        """Append an event. Must not raise for well-formed events."""
        ...

    def events_for(self, publication_run_id: str) -> list[PublicationEvent]:
        """Return all events for one publication run, in append order."""
        ...


class InMemoryEventLog:
    """In-memory PublicationEventLog; the launch implementation.

    A DB-backed store (append-only table, publisher-owned) replaces this
    without changing the publisher.
    """

    def __init__(self) -> None:
        self._events: list[PublicationEvent] = []
        self._counter = 0

    def append(self, event: PublicationEvent) -> None:
        self._counter += 1
        stamped = PublicationEvent(
            event_id=f"evt-{self._counter:08d}",
            publication_run_id=event.publication_run_id,
            product=event.product,
            audience=event.audience,
            kind=event.kind,
            release_id=event.release_id,
            manifest_hash=event.manifest_hash,
            detail=event.detail,
            timestamp=event.timestamp,
        )
        self._events.append(stamped)

    def events_for(self, publication_run_id: str) -> list[PublicationEvent]:
        return [e for e in self._events if e.publication_run_id == publication_run_id]

    @property
    def all_events(self) -> Sequence[PublicationEvent]:
        return tuple(self._events)
