"""Lanceur : exécute chaque connecteur actif et rassemble les Items (pipeline demo → contrat A).

Un connecteur qui plante n'empêche pas les autres : l'erreur est journalisée et
ajoutée aux warnings. Le message d'erreur n'est jamais recopié, il pourrait contenir
une adresse ou un jeton : seuls le nom du connecteur et le type d'erreur sont notés.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from contracts.models import Item
from packages.collect.core.connector import Connector, Window
from packages.collect.core.normalize import deduplicate
from packages.collect.core.registry import enabled_connectors

logger = logging.getLogger("collect")


@dataclass
class CollectResult:
    items: list[Item] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run(user: str, window: Window, connectors: list[Connector] | None = None) -> CollectResult:
    """Collecte tout pour `user` sur `window` avec les connecteurs donnés (par défaut : les actifs)."""
    result = CollectResult()
    for connector in enabled_connectors() if connectors is None else connectors:
        try:
            items = list(connector.collect(user, window))
        except Exception as exc:  # noqa: BLE001 — isoler chaque source est le but
            logger.error("connecteur %s en échec : %s", connector.name, type(exc).__name__)
            result.warnings.append(f"{connector.name}: échec ({type(exc).__name__})")
            continue
        for item in items:
            if not isinstance(item, Item) or item.owner_user_id != user:
                result.warnings.append(
                    f"{connector.name}: élément rejeté (hors contrat ou autre utilisateur)"
                )
                continue
            result.items.append(item)
        logger.info("connecteur %s : %d Items", connector.name, len(items))
    collected = len(result.items)
    result.items = deduplicate(result.items)  # L1.6 : un seul Item par item_id
    if len(result.items) < collected:
        result.warnings.append(f"{collected - len(result.items)} doublon(s) retiré(s)")
    return result
