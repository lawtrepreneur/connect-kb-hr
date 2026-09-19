"""Migration runner — connect-kb-hr#1.

Applies SQL migration files from migrations/ in order against a target DSN.
Idempotent: tracks applied migrations in a _migrations table in the public schema.

Usage:
    python3 -m connect_kb_hr.db.migrate --dsn postgresql://... up
    python3 -m connect_kb_hr.db.migrate --dsn postgresql://... down  # last migration only
    CORPUS_HR_DSN=postgresql://... python3 -m connect_kb_hr.db.migrate up
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "migrations"


def _get_dsn(args) -> str:
    dsn = getattr(args, "dsn", None) or os.environ.get("CORPUS_HR_DSN", "")
    if not dsn:
        print("ERROR: --dsn or CORPUS_HR_DSN required", file=sys.stderr)
        sys.exit(1)
    return dsn


def _ensure_migrations_table(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public._migrations (
            name        TEXT        NOT NULL PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def _applied(cur) -> set[str]:
    cur.execute("SELECT name FROM public._migrations")
    return {row[0] for row in cur.fetchall()}


def run_up(dsn: str) -> None:
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed", file=sys.stderr)
        sys.exit(1)

    migration_files = sorted(
        f for f in MIGRATIONS_DIR.glob("*.sql") if not f.name.endswith("_rollback.sql")
    )

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            _ensure_migrations_table(cur)
            conn.commit()
            applied = _applied(cur)

        for path in migration_files:
            if path.name in applied:
                print(f"  skip  {path.name} (already applied)")
                continue
            print(f"  apply {path.name} ...", end=" ", flush=True)
            sql = path.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO public._migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (path.name,),
                )
            conn.commit()
            print("ok")

    print("Migrations complete.")


def run_down(dsn: str) -> None:
    """Roll back the most recently applied migration."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed", file=sys.stderr)
        sys.exit(1)

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            _ensure_migrations_table(cur)
            cur.execute("SELECT name FROM public._migrations ORDER BY applied_at DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("Nothing to roll back.")
                return
            last = row[0]
            rollback_name = last.replace(".sql", "_rollback.sql")
            rollback_path = MIGRATIONS_DIR / rollback_name
            if not rollback_path.exists():
                print(f"ERROR: no rollback file found for {last}: expected {rollback_path}", file=sys.stderr)
                sys.exit(1)
            print(f"  rollback {last} ...", end=" ", flush=True)
            cur.execute(rollback_path.read_text())
            cur.execute("DELETE FROM public._migrations WHERE name = %s", (last,))
        conn.commit()
        print("ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="connect-kb-hr migration runner")
    parser.add_argument("--dsn", help="PostgreSQL DSN (default: CORPUS_HR_DSN env var)")
    parser.add_argument("direction", choices=["up", "down"], help="up: apply; down: rollback last")
    args = parser.parse_args()
    dsn = _get_dsn(args)
    if args.direction == "up":
        run_up(dsn)
    else:
        run_down(dsn)


if __name__ == "__main__":
    main()
