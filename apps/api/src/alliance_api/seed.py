"""Amorçage des données de démonstration (L3.8, P3).

`python -m alliance_api.seed [--data-dir DIR]` rejoue la collecte du matin du lot 1 avec
son connecteur `demo` pour chaque journée factice de `contracts/fixtures/demo` qui n'est pas
encore dans `DATA_DIR` : Items (`<user>/<jour>/items.jsonl`, sans corps complet si la
rétention l'exige) et journal de collecte (`<user>/audit.jsonl`), horodaté à 8 h 30 le jour
de la collecte comme une vraie collecte planifiée.

Sans effet sur une journée déjà présente : relancé à chaque démarrage du conteneur, il ne
réécrit rien et ne duplique pas le journal. Aucun compte ni réseau : données synthétiques.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from alliance_api.auth import valid_user_id
from contracts.demo_data import DEMO_DIR, available_days
from packages.collect.connectors.demo_fixtures import DemoFixturesConnector
from packages.collect.core.audit import AuditLog
from packages.collect.core.connector import collection_window
from packages.collect.core.retention import load_policy, without_content
from packages.collect.core.runner import run
from packages.collect.core.store import day_path, write_day


def seed(data_dir: Path, source: Path = DEMO_DIR) -> int:
    """Collecte les journées factices absentes de `data_dir` ; renvoie leur nombre."""
    connector = DemoFixturesConnector(source)
    audit = AuditLog(data_dir)
    policy = load_policy()
    added = 0
    users = sorted(p.name for p in Path(source).iterdir() if p.is_dir() and valid_user_id(p.name))
    for user in users:
        for name in available_days(user, source):
            day = date.fromisoformat(name)
            if day_path(data_dir, user, day).exists():
                continue
            window = collection_window(day)
            result = run(user, window, [connector], audit=audit.write, clock=lambda w=window: w.end)
            write_day(without_content(result.items, policy), data_dir, user, day)
            added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Amorçage des données de démonstration")
    parser.add_argument("--data-dir", help="dossier des Items (par défaut : variable DATA_DIR)")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if not data_dir:
        print(
            "amorçage ignoré : DATA_DIR vide (journées de démonstration intégrées)", file=sys.stderr
        )
        return 0
    added = seed(Path(data_dir))
    print(f"amorçage : {added} journée(s) ajoutée(s) dans {data_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
