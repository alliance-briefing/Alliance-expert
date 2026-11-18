"""Connecteur fichiers OneDrive : créés ou modifiés dans les 48 h (guide L1.5).

Deux sources au choix (variable FILES_SOURCE) :
- « files » (par défaut) : fichiers fabriqués (tests), servis uniquement à l'utilisateur de
  démonstration DEMO_USER (isolation L1.7) ;
- « graph » : le vrai compte, permission Files.Read (lecture seule).

Règles :
- fichiers créés ou modifiés dans les 48 h avant l'heure de collecte, 100 au plus (les plus
  récents) ; dossiers et éléments supprimés ignorés ;
- extrait du contenu pour .txt, .md, .csv (seuls les 8 premiers Ko sont lus) et .docx
  (lu seulement s'il fait moins de 2 Mo) ; les autres types (pdf…) : métadonnées seules ;
- un fichier devenu inaccessible (accès retiré, supprimé entre-temps) ne produit pas d'Item.

Côté Graph : `/me/drive/root/delta` liste tout le OneDrive en quelques pages ; le contenu est
lu par le lien de téléchargement pré-signé du fichier, avec un en-tête Range, sans jeton.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

from contracts.models import Item
from packages.collect.connectors.files_graph.mapping import (
    DOCX,
    TEXT_TYPES,
    content_text,
    extension,
    file_to_item,
)
from packages.collect.core.auth import get_access_token
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.graph_http import GraphError, Send, get_all, get_one, urllib_send
from packages.collect.core.normalize import normalize_all
from packages.collect.core.paris_time import parse_graph

logger = logging.getLogger("collect.files_graph")

RECENT = timedelta(hours=48)
MAX_FILES = 100
TEXT_BYTES = 8 * 1024  # assez pour un extrait de 500 caractères
DOCX_MAX_SIZE = 2 * 1024 * 1024  # un docx se lit en entier : au-delà, métadonnées seules
DEMO_USER = "u_demo_graph"
DEFAULT_DIR = Path(
    os.environ.get("FILES_GRAPH_DIR", Path(__file__).parents[2] / "tests/fixtures/graph/files")
)


class Gone(Exception):
    """Le fichier n'est plus accessible (accès retiré ou supprimé) : pas d'Item."""


@dataclass(frozen=True)
class DriveSource:
    files: Callable[[str], list[dict]]  # utilisateur → driveItems
    read: Callable[[str, dict, int | None], bytes]  # (utilisateur, fichier, octets max) → contenu


def file_source(folder: Path = DEFAULT_DIR, owner: str = DEMO_USER) -> DriveSource:
    """delta.json (driveItems) + content/<id> (contenus). Données de `owner` seul."""

    def files(user: str) -> list[dict]:
        if user != owner:
            return []
        return json.loads((Path(folder) / "delta.json").read_text("utf-8"))["value"]

    def read(user: str, item: dict, limit: int | None) -> bytes:
        content = Path(folder) / "content" / item["id"]
        if user != owner or not content.exists():
            raise Gone
        data = content.read_bytes()
        return data[:limit] if limit else data

    return DriveSource(files, read)


def graph_source(
    token_for: Callable[[str], str] = get_access_token, send: Send = urllib_send
) -> DriveSource:
    def files(user: str) -> list[dict]:
        return get_all("/me/drive/root/delta", token_for(user), send=send)

    def read(user: str, item: dict, limit: int | None) -> bytes:
        try:
            meta = get_one(
                f"/me/drive/items/{quote(item['id'], safe='')}", token_for(user), send=send
            )
        except GraphError as error:
            if error.status in (403, 404, 410):
                raise Gone from error
            raise
        url = meta.get("@microsoft.graph.downloadUrl")
        if not url:
            raise Gone
        headers = {"Range": f"bytes=0-{limit - 1}"} if limit else {}
        status, _, body = send(url, headers)  # lien pré-signé : surtout pas de jeton
        if status in (403, 404, 410):
            raise Gone
        if status not in (200, 206):
            raise GraphError(status)
        return body[:limit] if limit else body

    return DriveSource(files, read)


class GraphFilesConnector(Connector):
    name = "files_graph"
    version = "0.1.0"

    def __init__(self, source: DriveSource | None = None) -> None:
        if source is None:
            source = graph_source() if os.environ.get("FILES_SOURCE") == "graph" else file_source()
        self.source = source

    def required_scopes(self) -> list[str]:
        return ["Files.Read"]  # lecture seule (guide L1.2)

    def _recent(self, user: str, window: Window) -> list[dict]:
        since, now = window.end - RECENT, window.end
        recent = [
            f
            for f in self.source.files(user)
            if "file" in f
            and "deleted" not in f
            and since < parse_graph(f["lastModifiedDateTime"]) <= now
        ]
        recent.sort(key=lambda f: f["lastModifiedDateTime"], reverse=True)
        if len(recent) > MAX_FILES:
            logger.warning(
                "%s : %d fichier(s) au-delà du plafond", self.name, len(recent) - MAX_FILES
            )
        return recent[:MAX_FILES]

    def _text(self, user: str, item: dict) -> str:
        """Extrait du contenu ; lève Gone si le fichier n'est plus accessible."""
        kind = extension(item)
        if kind in TEXT_TYPES:
            return content_text(item, self.source.read(user, item, TEXT_BYTES))
        if kind == DOCX and int(item.get("size") or 0) <= DOCX_MAX_SIZE:
            return content_text(item, self.source.read(user, item, None))
        self.source.read(user, item, 1)  # métadonnées seules, mais l'accès est vérifié
        return ""

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        readable: list[dict] = []
        for item in self._recent(user, window):
            try:
                readable.append({"item": item, "text": self._text(user, item)})
            except Gone:
                logger.info("%s : fichier devenu inaccessible, ignoré", self.name)
        result = normalize_all(
            readable,
            lambda raw: file_to_item(raw["item"], user, window.end, raw["text"]),
            connector=self.name,
        )  # un fichier invalide est rejeté seul (L1.6)
        yield from result.items

    def health(self) -> Health:
        try:
            self.source.files("health")
        except Exception as exc:  # noqa: BLE001
            return Health(False, type(exc).__name__)
        return Health(True, "source joignable")
