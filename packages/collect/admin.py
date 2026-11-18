"""Commandes d'administration des données du lot 1 (guide L1.7).

  uv run python -m packages.collect.admin delete-user --user u_demo_001          # aperçu
  uv run python -m packages.collect.admin delete-user --user u_demo_001 --yes    # suppression
  uv run python -m packages.collect.admin purge [--today AAAA-MM-JJ]             # rétention

Le dossier vient de --data-dir ou de la variable DATA_DIR. delete-user efface aussi les jetons
OAuth de l'utilisateur (TOKENS_DIR) : plus aucune collecte possible sans nouvelle connexion.
Sans --yes, delete-user ne fait qu'afficher ce qui serait supprimé : une suppression ne se
rattrape pas.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, date, datetime

from packages.collect.core.oauth import TokenStore
from packages.collect.core.paris_time import to_paris
from packages.collect.core.retention import delete_user, load_policy, purge
from packages.collect.core.store import user_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Données du lot 1 : suppression et rétention")
    parser.add_argument("--data-dir", help="dossier des données (par défaut : variable DATA_DIR)")
    commands = parser.add_subparsers(dest="command", required=True)
    delete = commands.add_parser("delete-user", help="effacer toutes les données d'un utilisateur")
    delete.add_argument("--user", required=True)
    delete.add_argument("--yes", action="store_true", help="supprimer vraiment")
    clean = commands.add_parser("purge", help="appliquer la politique de rétention")
    clean.add_argument("--today", help="jour de référence AAAA-MM-JJ (par défaut : aujourd'hui)")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or os.environ.get("DATA_DIR")
    if not data_dir:
        parser.error("--data-dir ou la variable DATA_DIR est obligatoire")

    if args.command == "delete-user":
        folder = user_folder(data_dir, args.user)
        tokens = TokenStore()
        has_tokens = tokens.path(args.user).exists()
        if not args.yes:
            count = sum(1 for p in folder.rglob("*") if p.is_file()) if folder.exists() else 0
            print(
                f"aperçu : {count} fichier(s) dans {folder}"
                f"{' + les jetons OAuth' if has_tokens else ''} ; relancer avec --yes pour supprimer"
            )
            return 0
        removed = delete_user(data_dir, args.user)
        tokens_removed = tokens.delete(args.user)
        print(
            f"{args.user} : {len(removed)} fichier(s) supprimé(s)"
            f"{', jetons OAuth effacés' if tokens_removed else ''}"
        )
        if tokens_removed:
            print(
                "Pour retirer aussi l'accord côté Microsoft : https://account.live.com/consent/Manage",
                file=sys.stderr,
            )
        print("À faire aussi : redémarrer l'API (cache des briefs en mémoire).", file=sys.stderr)
        return 0

    today = date.fromisoformat(args.today) if args.today else to_paris(datetime.now(UTC)).date()
    report = purge(data_dir, today, load_policy())
    print(
        f"rétention : {len(report.days_removed)} journée(s) supprimée(s), "
        f"{report.audit_lines_removed} ligne(s) d'audit retirée(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
