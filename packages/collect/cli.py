"""Lance la collecte d'une journée et affiche les Items (une ligne JSON par Item).

Usage : uv run python -m packages.collect.cli --user u_demo_001 --day 2026-10-05 [--source demo]

Avec --write, les Items sont écrits dans <DATA_DIR>/<user>/<jour>/items.jsonl (lu par l'API)
au lieu d'être affichés ; le dossier vient de --data-dir ou de la variable DATA_DIR.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

from packages.collect.core.connector import collection_window
from packages.collect.core.registry import discover
from packages.collect.core.runner import run
from packages.collect.core.store import write_day


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collecte du lot 1 (contrat A)")
    parser.add_argument("--user", required=True, help="identifiant de l'utilisateur")
    parser.add_argument("--day", required=True, help="jour de collecte, AAAA-MM-JJ (à 8h30, Paris)")
    parser.add_argument("--source", help="ne lancer que ce connecteur, même s'il est désactivé")
    parser.add_argument("--write", action="store_true", help="écrire items.jsonl dans DATA_DIR")
    parser.add_argument("--data-dir", help="dossier des Items (par défaut : variable DATA_DIR)")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if args.write and not data_dir:
        parser.error("--write demande --data-dir ou la variable DATA_DIR")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    day = date.fromisoformat(args.day)
    connectors = [discover()[args.source]()] if args.source else None
    result = run(args.user, collection_window(day), connectors)
    if args.write:
        print(f"écrit : {write_day(result.items, data_dir, args.user, day)}", file=sys.stderr)
    else:
        for item in result.items:
            print(item.model_dump_json())
    for warning in result.warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    print(f"{len(result.items)} Items, {len(result.warnings)} warning(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
