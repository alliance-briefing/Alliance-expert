"""Source n°5 de démonstration (preuve L1.1) : des notes Markdown factices.

Chaque fichier notes/AAAA-MM-JJ-titre.md devient un Item `source = "file"`.
La première ligne (« # Titre ») donne le titre, le reste donne l'extrait.
Sont collectées les notes des 2 jours précédant la journée demandée, et celle du jour.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from contracts.models import Item
from packages.collect.core.connector import Connector, Health, Window

NOTES_DIR = Path(__file__).parent / "notes"
SNIPPET_MAX = 500


class NotesDemoConnector(Connector):
    name = "notes_demo"
    version = "0.1.0"

    def __init__(self, notes_dir: Path = NOTES_DIR) -> None:
        self.notes_dir = Path(notes_dir)

    def required_scopes(self) -> list[str]:
        return []  # notes locales et factices : aucune autorisation

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        today = window.start.date()
        for path in sorted(self.notes_dir.glob("*.md")):
            day = date.fromisoformat(path.name[:10])
            if not today - timedelta(days=2) <= day <= today:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            title = lines[0].lstrip("# ").strip() if lines else path.stem
            body = " ".join(line.strip() for line in lines[1:] if line.strip())
            written = datetime.combine(day, time(9, 0), UTC)
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
