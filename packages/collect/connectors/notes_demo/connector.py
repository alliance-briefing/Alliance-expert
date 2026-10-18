"""Source n°5 de démonstration (preuve L1.1) : des notes Markdown factices.

Chaque fichier notes/AAAA-MM-JJ-titre.md devient un Item `source = "file"`, daté du jour
de la note à 9h00 (heure de Paris). La première ligne (« # Titre ») donne le titre,
le reste donne l'extrait.

Isolation (L1.7) : les notes sont des données de démonstration. Elles ne sont servies
qu'aux utilisateurs de la démo (ceux qui ont un dossier dans contracts/fixtures/demo/),
et seulement si elles tombent dans la fenêtre de collecte demandée.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

from contracts.demo_data import DEMO_DIR
from contracts.models import Item
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.paris_time import paris_time

NOTES_DIR = Path(__file__).parent / "notes"
SNIPPET_MAX = 500
NOTE_HOUR = 9  # une note est réputée écrite à 9h00, heure de Paris


def demo_users(root: Path = DEMO_DIR) -> set[str]:
    """Utilisateurs de la démo : un dossier par utilisateur dans les fixtures."""
    return {p.name for p in Path(root).iterdir() if p.is_dir()} if Path(root).is_dir() else set()


class NotesDemoConnector(Connector):
    name = "notes_demo"
    version = "0.1.0"

    def __init__(self, notes_dir: Path = NOTES_DIR, users: set[str] | None = None) -> None:
        self.notes_dir = Path(notes_dir)
        self.users = demo_users() if users is None else users

    def required_scopes(self) -> list[str]:
        return []  # notes locales et factices : aucune autorisation

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        if user not in self.users:
            return  # utilisateur inconnu de la démo : aucune donnée
        for path in sorted(self.notes_dir.glob("*.md")):
            written = paris_time(date.fromisoformat(path.name[:10]), NOTE_HOUR)
            if not window.start <= written < window.end:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            title = lines[0].lstrip("# ").strip() if lines else path.stem
            body = " ".join(line.strip() for line in lines[1:] if line.strip())
            yield Item(
                item_id=f"file:note-{path.stem}",
                source="file",
                provider="demo",
                owner_user_id=user,
                title=title,
                snippet=body[:SNIPPET_MAX],
                created_at=written,
                updated_at=written,
                url=f"https://notes.demo.example/{path.stem}",
                acl={"visible_to": [user]},
                collected_at=window.end,
                connector={"name": self.name, "version": self.version},
            )

    def health(self) -> Health:
        count = len(list(self.notes_dir.glob("*.md"))) if self.notes_dir.is_dir() else 0
        return Health(count > 0, f"{count} note(s)")
