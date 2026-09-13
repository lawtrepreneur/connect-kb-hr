"""Product-owned publication targets (issue #20, ADR-0005 sections 8, 15).

Each product (LegalX, SignalX, HR) owns its own PostgreSQL database and its
own publisher credentials. A TargetConfig binds one audience schema inside
one product database to one publisher role. Credentials are
environment-variable-only and never logged.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class TargetConfig:
    """One product-owned publication target.

    ``product`` is the owning product (e.g. ``legalx``, ``signalx``, ``hr``),
    ``audience`` the audience corpus inside that product (e.g. ``employer``,
    ``employee``, ``public``), ``schema_name`` the audience schema in the
    product database, and ``publisher_role`` the least-privilege role the
    publisher connects as. ``dsn`` is the PostgreSQL connection string for
    that role; it is never logged or serialized.
    """

    product: str
    audience: str
    dsn: str
    publisher_role: str
    schema_name: str

    def __post_init__(self) -> None:
        if not self.product:
            raise ValueError("product is required")
        if not self.audience:
            raise ValueError("audience is required")
        if not self.dsn:
            raise ValueError("dsn is required (never logged)")
        if not self.publisher_role:
            raise ValueError("publisher_role is required")
        if not _IDENT_RE.match(self.schema_name):
            raise ValueError(f"schema_name must be a lowercase SQL identifier: {self.schema_name!r}")

    @classmethod
    def from_env(cls, product: str, audience: str) -> "TargetConfig":
        """Build a target from environment variables (AuthConfig convention).

        Reads ``CORPUS_<PRODUCT>_<AUDIENCE>_DSN`` and
        ``CORPUS_<PRODUCT>_<AUDIENCE>_PUBLISHER_ROLE``; the schema defaults
        to ``<product>_<audience>``.
        """
        base = f"CORPUS_{product.upper()}_{audience.upper()}"
        dsn = os.environ.get(f"{base}_DSN", "")
        role = os.environ.get(f"{base}_PUBLISHER_ROLE", "")
        if not dsn:
            raise ValueError(f"{base}_DSN is required (no default)")
        if not role:
            raise ValueError(f"{base}_PUBLISHER_ROLE is required (no default)")
        return cls(
            product=product,
            audience=audience,
            dsn=dsn,
            publisher_role=role,
            schema_name=f"{product}_{audience}",
        )

    def describe(self) -> str:
        """Log-safe description: product, audience, role — never the DSN."""
        return f"target(product={self.product}, audience={self.audience}, role={self.publisher_role}, schema={self.schema_name})"
