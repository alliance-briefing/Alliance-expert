"""Lance la collecte d'une journée et affiche les Items (une ligne JSON par Item).

Usage : uv run python -m packages.collect.cli --user u_demo_001 --day 2026-10-05 [--source demo]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from packages.collect.core.connector import collection_window
from packages.collect.core.registry import discover
from packages.collect.core.runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collecte du lot 1 (contrat A)")
    parser.add_argument("--user", required=True, help="identifiant de l'utilisateur")
    parser.add_argument("--day", required=True, help="jour de collecte, AAAA-MM-JJ (à 8h30, Paris)")
    parser.add_argument("--source", help="ne lancer que ce connecteur, même s'il est désactivé")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    window = collection_window(date.fromisoformat(args.day))
    connectors = [discover()[args.source]()] if args.source else None
    result = run(args.user, window, connectors)
    for item in result.items:
        print(item.model_dump_json())
    for warning in result.warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    print(f"{len(result.items)} Items, {len(result.warnings)} warning(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
