"""Écrit la collecte d'une journée là où l'API la lit : DATA_DIR/<user>/<AAAA-MM-JJ>/items.jsonl.

Même format que les journées de démonstration (contracts/fixtures/demo) : un Item du
contrat A par ligne, lu par `contracts.demo_data.load_items` (README, « Brancher d'autres
données »). Le dossier du jour porte la date du matin de collecte (8h30, Paris).
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from contracts.models import Item

FILE_NAME = "items.jsonl"
# L'identifiant devient un nom de dossier : pas de « / » ni de « .. » (isolation L1.7).
_SAFE_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def day_path(root: Path | str, user: str, day: date) -> Path:
    if not _SAFE_USER.fullmatch(user):
        raise ValueError("identifiant utilisateur invalide pour un nom de dossier")
    return Path(root) / user / day.isoformat() / FILE_NAME


def write_day(items: Iterable[Item], root: Path | str, user: str, day: date) -> Path:
    """Remplace le fichier du jour (jamais d'ajout à la suite) et renvoie son chemin.

    Une collecte vide écrit un fichier vide : l'API sait alors que la journée existe.
    L'écriture passe par un fichier temporaire : l'API ne lit jamais un fichier à moitié écrit.
    """
    path = day_path(root, user, day)
    items = sorted(items, key=lambda item: item.item_id)  # même collecte = même fichier
    if any(item.owner_user_id != user for item in items):
        raise ValueError("un Item appartient à un autre utilisateur")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".items-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(item.model_dump_json() + "\n" for item in items)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
