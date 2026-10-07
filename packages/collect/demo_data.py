"""Lecture des journées factices (fixtures du lot 1) — tout ce dont le lot 2 a besoin, sans dépendre du lot 1.

Exemple (stub de brief L2.0) :

    from contracts.demo_data import available_days, load_items

    day = available_days()[0]                      # "2026-10-05"
    items = load_items(day)                        # list[Item], déjà validés par le contrat A
    brief = generate_brief(items, "u_demo_001", day)

La vérité terrain (`load_truth`) est réservée à l'évaluation (lot 3) : le lot 2 ne doit pas
l'utiliser pour régler ses règles, sinon l'évaluation n'est plus indépendante (guide L3.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts.models import Item

DEMO_DIR = Path(__file__).parent / "fixtures" / "demo"
DEFAULT_USER = "u_demo_001"


def available_days(user_id: str = DEFAULT_USER, root: Path = DEMO_DIR) -> list[str]:
    """Jours disponibles (AAAA-MM-JJ), triés."""
    folder = Path(root) / user_id
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if (p / "items.jsonl").exists())


def load_items(day: str, user_id: str = DEFAULT_USER, root: Path = DEMO_DIR) -> list[Item]:
    """Les Item collectés le matin du jour `day` (contrat A), validés à la lecture."""
    path = Path(root) / user_id / day / "items.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [Item.model_validate_json(line) for line in handle if line.strip()]


def load_truth(day: str, user_id: str = DEFAULT_USER, root: Path = DEMO_DIR) -> dict:
    """Vérité terrain de la journée : réservée à l'évaluation (lot 3)."""
    return json.loads((Path(root) / user_id / day / "truth.json").read_text(encoding="utf-8"))
