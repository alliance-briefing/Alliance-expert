"""Connecteur « demo » : renvoie les Items d'une journée factice au lieu d'appeler une API.

Il relit contracts/fixtures/demo/ : la journée choisie est celle de la collecte (window.end, heure de Paris).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from contracts.demo_data import DEMO_DIR, available_days, load_items
from contracts.models import Item
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.paris_time import to_paris


class DemoFixturesConnector(Connector):
    name = "demo"
    version = "0.2.0"

    def __init__(self, root: Path = DEMO_DIR) -> None:
        self.root = Path(root)

    def required_scopes(self) -> list[str]:
        return []  # données locales et factices : aucune autorisation

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        day = to_paris(window.end).date().isoformat()
        if day not in available_days(user, self.root):
            return
        for item in load_items(day, user, self.root):
            if user in item.acl.visible_to:
                yield item

    def health(self) -> Health:
        if not self.root.exists():
            return Health(False, f"dossier absent : {self.root.name}")
        days = sum(
            len(available_days(p.name, self.root)) for p in self.root.iterdir() if p.is_dir()
        )
        return Health(days > 0, f"{days} journée(s) disponible(s)")
