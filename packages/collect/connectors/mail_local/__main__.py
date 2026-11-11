"""Lancer : python -m packages.collect.connectors.mail_local --date 2026-10-21."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, timedelta
from pathlib import Path

from packages.collect.connectors.mail_local.connector import DEFAULT_FIXTURES, LocalMailConnector
from packages.collect.core.connector import Window
from packages.collect.synth.clock import collect_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partie B : préparer les mails du jeu de test local"
    )
    parser.add_argument(
        "--date", type=date.fromisoformat, required=True, help="date simulée AAAA-MM-JJ"
    )
    parser.add_argument("--user", default="u_demo_001")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=Path("sorties/mails.json"))
    args = parser.parse_args()
    connector = LocalMailConnector(args.fixtures)
    if connector.health().status == "down":
        parser.error("manifest.json absent du dossier de test")
    now = collect_time(args.date)
    window = Window(now.astimezone(UTC) - timedelta(hours=24), now)
    try:
        items = list(connector.collect(args.user, window))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    recent_start = now.astimezone(UTC) - timedelta(hours=24)
    recent_count = sum(item.created_at.astimezone(UTC) > recent_start for item in items)
    print(f"Reference : {now.isoformat()}")
    print(
        f"Mails retenus : {len(items)} ({recent_count} recents, {len(items) - recent_count} anciens sans reponse)"
    )
    print(f"Fichier : {args.out.resolve()}")
    for item in items[:3]:
        sender = next(p.email for p in item.participants if p.role == "from")
        # JSON ASCII uniquement dans le terminal Windows ; fichier de sortie en UTF-8.
        print(
            json.dumps(
                {
                    "item_id": item.item_id,
                    "from": sender,
                    "title": item.title,
                    "snippet": item.snippet,
                }
            )
        )


if __name__ == "__main__":
    main()
