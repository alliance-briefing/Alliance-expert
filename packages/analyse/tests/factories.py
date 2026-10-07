"""Fabrique d'Item synthétiques pour les tests du lot 2.

Chaque fonction produit un Item valide du contrat A avec des valeurs par défaut ;
un test ne précise que les champs qui l'intéressent. Données fictives uniquement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from contracts.models import Item

USER = "u_demo_001"
OTHER_USER = "u_demo_002"
COLLECTED_AT = "2026-10-07T08:30:02+02:00"


def make_item(source: str, native_id: str, **fields: Any) -> Item:
    owner = fields.pop("owner_user_id", USER)
    data: dict[str, Any] = {
        "item_id": f"{source}:{native_id}",
        "source": source,
        "provider": "demo",
        "owner_user_id": owner,
        "title": f"Titre {native_id}",
        "snippet": "",
        "created_at": "2026-10-01T09:00:00+02:00",
        "updated_at": "2026-10-01T09:00:00+02:00",
        "url": f"https://demo.reseau-expertis.test/{source}/{native_id}",
        "acl": {"visible_to": [owner]},
        "collected_at": COLLECTED_AT,
        "connector": {"name": "demo", "version": "0.1.0"},
    }
    data.update(fields)
    return Item.model_validate(data)


def make_mail(native_id: str, created_at: str, *, is_read: bool = False, **fields: Any) -> Item:
    return make_item(
        "mail", native_id, created_at=created_at, updated_at=created_at, is_read=is_read, **fields
    )


def make_event(native_id: str, start_at: str, end_at: str, **fields: Any) -> Item:
    return make_item("calendar", native_id, start_at=start_at, end_at=end_at, **fields)


def make_task(
    native_id: str, due_at: str | None, *, completed: bool = False, **fields: Any
) -> Item:
    return make_item("task", native_id, due_at=due_at, completed=completed, **fields)


def at(value: str) -> datetime:
    """Raccourci lisible pour un instant ISO 8601 avec décalage."""
    return datetime.fromisoformat(value)
