"""Traduction d'un fichier OneDrive (Graph `driveItem`) en Item du contrat A 1.1.

Le contrat 1.1 n'a pas de bloc `attributes` pour les fichiers : le chemin, le type, la taille
et le dernier modificateur sont donc écrits en tête de l'extrait, une ligne lisible par le
lot 2 (« Altaïr/Écarts achats T3.csv · CSV · 126 o · modifié par Claire Fontaine »), suivie
de l'extrait du contenu quand il y en a un. Le dernier modificateur est aussi un
participant (`modifier`) quand Graph donne son adresse.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

from contracts.models import Item
from packages.collect.core.paris_time import parse_graph, to_paris
from packages.collect.core.text import html_to_text, truncate

CONNECTOR = {"name": "files_graph", "version": "0.1.0"}
TEXT_TYPES = {".txt", ".md", ".csv"}  # extrait lu sur les premiers octets
DOCX = ".docx"  # extrait lu dans le zip : fichier entier, donc petit seulement
_SPACES = re.compile(r"\s+")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _one_line(text: str) -> str:
    return _SPACES.sub(" ", "".join(c if c.isprintable() else " " for c in text)).strip()


def path(drive_item: dict) -> str:
    """Chemin lisible depuis la racine du OneDrive : « Altaïr/Écarts achats T3.csv »."""
    parent = (drive_item.get("parentReference") or {}).get("path", "")
    folder = parent.split("root:", 1)[1].strip("/") if "root:" in parent else ""
    return f"{folder}/{drive_item['name']}" if folder else drive_item["name"]


def extension(drive_item: dict) -> str:
    return PurePosixPath(drive_item["name"]).suffix.lower()


def human_size(size: int) -> str:
    for unit, step in (("Go", 1 << 30), ("Mo", 1 << 20), ("Ko", 1 << 10)):
        if size >= step:
            return f"{size / step:.1f} {unit}".replace(".0 ", " ")
    return f"{size} o"


def modifier(drive_item: dict) -> tuple[str, str | None]:
    who = (drive_item.get("lastModifiedBy") or {}).get("user") or {}
    email = who.get("email")
    return who.get("displayName") or "", email if email and _EMAIL.match(email) else None


def content_text(drive_item: dict, data: bytes) -> str:
    """Texte brut d'un contenu lu (texte ou docx) ; vide si illisible."""
    if extension(drive_item) == DOCX:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xml = archive.read("word/document.xml").decode("utf-8", "ignore")
        except (zipfile.BadZipFile, KeyError):
            return ""
        return html_to_text(re.sub(r"</w:p>", "\n", xml))
    return data.decode("utf-8", "ignore")


def file_to_item(drive_item: dict, owner: str, collected_at: datetime, text: str = "") -> Item:
    name, email = modifier(drive_item)
    kind = extension(drive_item).lstrip(".").upper() or "fichier"
    header = " · ".join(
        part
        for part in (
            path(drive_item),
            kind,
            human_size(int(drive_item.get("size") or 0)),
            f"modifié par {name}" if name else "",
        )
        if part
    )
    created = to_paris(parse_graph(drive_item["createdDateTime"]))
    updated = max(created, to_paris(parse_graph(drive_item["lastModifiedDateTime"])))
    participants = [{"name": name or email, "email": email, "role": "modifier"}] if email else []
    return Item(
        item_id=f"file:{drive_item['id']}",
        source="file",
        provider="microsoft",
        owner_user_id=owner,
        title=_one_line(drive_item["name"]),
        snippet=truncate(_one_line(f"{header} — {text}" if text.strip() else header)),
        created_at=created,
        updated_at=updated,
        participants=participants,
        url=drive_item["webUrl"],
        acl={"visible_to": [owner]},
        collected_at=to_paris(collected_at),
        connector=CONNECTOR,
    )
