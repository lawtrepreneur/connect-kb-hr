"""Product-owned publication targets (unified corpus design, ADR-0005 rev).

Each product (LegalX, SignalX, HR) owns its own PostgreSQL database and its
own publisher credentials. A TargetConfig binds one product to one unified
'kb' schema; audience is an assignment predicate, not a schema selector.
Credentials are environment-variable-only and never logged.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True, repr=False)
class TargetConfig:
    """One product-owned publication target.

    ``product`` is the owning product (e.g. ``legalx``, ``signalx``, ``hr``),
    ``schema_name`` the unified schema in the product database (defaults to
    ``'kb'``), and ``publisher_role`` the least-privilege role the publisher
    connects as. ``dsn`` is the PostgreSQL connection string for that role;
    it is never logged or serialized.

    Audience is an assignment predicate stored in ``kb.release_assignments``
    — it is NOT encoded in the target or schema name.
    """

    product: str
    dsn: str
    publisher_role: str
    schema_name: str = "kb"

    def __post_init__(self) -> None:
        if not self.product:
            raise ValueError("product is required")
        if not self.dsn:
            raise ValueError("dsn is required (never logged)")
        if not self.publisher_role:
            raise ValueError("publisher_role is required")
        if not _IDENT_RE.match(self.schema_name):
            raise ValueError(
                f"schema_name must be a lowercase SQL identifier: {self.schema_name!r}"
            )

    @classmethod
    def from_env(cls, product: str) -> "TargetConfig":
        """Build a target from environment variables.

        Reads ``CORPUS_<PRODUCT>_DSN`` and
        ``CORPUS_<PRODUCT>_PUBLISHER_ROLE``; the schema defaults to ``'kb'``.
        """
        base = f"CORPUS_{product.upper()}"
        dsn = os.environ.get(f"{base}_DSN", "")
        role = os.environ.get(f"{base}_PUBLISHER_ROLE", "")
        if not dsn:
            raise ValueError(f"{base}_DSN is required (no default)")
        if not role:
            raise ValueError(f"{base}_PUBLISHER_ROLE is required (no default)")
        return cls(
            product=product,
            dsn=dsn,
            publisher_role=role,
            schema_name="kb",
        )

    def __repr__(self) -> str:
        """DSN is omitted; use describe() for log-safe human output."""
        return (
            f"TargetConfig(product={self.product!r},"
            f" dsn=<redacted>, publisher_role={self.publisher_role!r},"
            f" schema_name={self.schema_name!r})"
        )

    def describe(self) -> str:
        """Log-safe description: product, role, schema — never the DSN."""
        return (
            f"target(product={self.product}, role={self.publisher_role},"
            f" schema={self.schema_name})"
        )
