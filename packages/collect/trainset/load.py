"""Chargeur du corpus d'entraînement, pour le lot 2 (qui ne génère rien lui-même).

Le corpus se génère d'abord : `uv run python -m packages.collect.trainset`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from contracts.models import Item
from packages.collect.trainset.build import DATA
from packages.collect.trainset.facts import label_on

__all__ = ["label_on", "load_items", "load_scenario", "personas"]


def _path(folder: str, key: str, data: Path) -> Path:
    path = data / folder / f"{key}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : lancer d'abord `uv run python -m packages.collect.trainset`"
        )
    return path


def personas(split: str | None = None, data: Path = DATA) -> list[str]:
    """Clés des personas du manifeste, filtrées par split (« train » ou « tgen »)."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    return [k for k, v in manifest["personas"].items() if split in (None, v["split"])]


def load_items(key: str, data: Path = DATA) -> list[Item]:
    text = _path("items", key, data).read_text(encoding="utf-8")
    return [Item.model_validate_json(line) for line in text.splitlines() if line]


def load_scenario(key: str, data: Path = DATA) -> list[dict]:
    text = _path("scenario", key, data).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def labels_at(key: str, moment: datetime, data: Path = DATA) -> dict[str, dict]:
    """Étiquettes des éléments visibles au moment du brief, par item_id."""
    out = {}
    for row in load_scenario(key, data):
        labels = label_on(row, moment)
        if labels is not None:
            out[row["item_id"]] = labels
    return out
