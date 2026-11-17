"""Connexion des comptes Microsoft (guide L1.2).

  uv run python -m packages.collect.auth keygen --save           # une fois : clé de chiffrement
  uv run python -m packages.collect.auth login  --user u_claire  # ouvre le navigateur
  uv run python -m packages.collect.auth status --user u_claire  # connecté ? jusqu'à quand ?
  uv run python -m packages.collect.auth logout --user u_claire  # efface les jetons

Les jetons sont chiffrés (Fernet) et rangés dans TOKENS_DIR (par défaut
~/.local/share/alliance-briefing/tokens). La clé vient de TOKEN_KEY, sinon du fichier
~/.config/alliance-briefing/token_key écrit par `keygen --save` (lisible par vous seul,
hors du dépôt). Aucun jeton n'est jamais affiché.
Pour retirer aussi l'accord côté Microsoft : https://account.live.com/consent/Manage
(compte personnel) ou « Mes applications » (compte professionnel).
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from cryptography.fernet import Fernet

from packages.collect.core.oauth import (
    DEFAULT_KEY_FILE,
    OAuthError,
    TokenStore,
    load_config,
    login,
    save_key,
)


def main(
    argv: list[str] | None = None, open_browser=webbrowser.open, key_file: Path = DEFAULT_KEY_FILE
) -> int:
    parser = argparse.ArgumentParser(description="Connexion des comptes Microsoft (OAuth)")
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="créer une clé de chiffrement des jetons")
    keygen.add_argument(
        "--save", action="store_true", help="la ranger dans ~/.config (sinon : afficher)"
    )
    for name, text in (
        ("login", "se connecter"),
        ("status", "état de la connexion"),
        ("logout", "effacer les jetons"),
    ):
        commands.add_parser(name, help=text).add_argument("--user", required=True)
    args = parser.parse_args(argv)

    if args.command == "keygen":
        key = Fernet.generate_key().decode()
        if not args.save:
            print(key)
            print("À mettre dans TOKEN_KEY (fichier .env non versionné).", file=sys.stderr)
            return 0
        try:
            save_key(key, key_file)
        except OAuthError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"clé rangée dans {key_file}")
        return 0

    store = TokenStore(key_file=key_file)
    try:
        if args.command == "login":
            print("Le navigateur s'ouvre sur la page de connexion de Microsoft…", file=sys.stderr)
            tokens = login(args.user, store, load_config(), open_browser)
            print(f"{args.user} : connecté, jeton valable jusqu'au {tokens.expires_at}")
        elif args.command == "status":
            tokens = store.load(args.user)
            print(f"{args.user} : connecté, jeton d'accès valable jusqu'au {tokens.expires_at}")
        else:
            removed = store.delete(args.user)
            print(f"{args.user} : {'jetons effacés' if removed else 'aucun jeton enregistré'}")
    except OAuthError as error:
        print(f"{args.user} : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
