"""Lanceur : exécute chaque connecteur actif et rassemble les Items (pipeline demo → contrat A).

Un connecteur qui plante n'empêche pas les autres : l'erreur est journalisée et
ajoutée aux warnings. Le message d'erreur n'est jamais recopié, il pourrait contenir
une adresse ou un jeton : seuls le nom du connecteur et le type d'erreur sont notés.

Avec `audit`, chaque connecteur laisse aussi une trace dans le journal d'audit (L1.8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from contracts.models import Item
from packages.collect.core.audit import AuditSink, Clock, collect_events, failure_event, utc_now
from packages.collect.core.connector import Connector, Window
from packages.collect.core.normalize import deduplicate
from packages.collect.core.registry import enabled_connectors

logger = logging.getLogger("collect")


@dataclass
class CollectResult:
    items: list[Item] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run(
    user: str,
    window: Window,
    connectors: list[Connector] | None = None,
    audit: AuditSink | None = None,
    clock: Clock = utc_now,
) -> CollectResult:
    """Collecte tout pour `user` sur `window` avec les connecteurs donnés (par défaut : les actifs).

    `audit` reçoit les AuditEvent de chaque connecteur (aucun journal si None).
    """
    result = CollectResult()
    for connector in enabled_connectors() if connectors is None else connectors:
        try:
            items = list(connector.collect(user, window))
        except Exception as exc:  # noqa: BLE001 — isoler chaque source est le but
            logger.error("connecteur %s en échec : %s", connector.name, type(exc).__name__)
            result.warnings.append(f"{connector.name}: échec ({type(exc).__name__})")
            if audit:
                audit(failure_event(connector.name, user, clock(), exc))
            continue
        accepted = []
        for item in items:
            if not isinstance(item, Item) or item.owner_user_id != user:
                result.warnings.append(
                    f"{connector.name}: élément rejeté (hors contrat ou autre utilisateur)"
                )
                continue
            accepted.append(item)
        result.items.extend(accepted)
        if audit:
            for event in collect_events(connector.name, user, accepted, clock()):
                audit(event)
        logger.info("connecteur %s : %d Items", connector.name, len(items))
    collected = len(result.items)
    result.items = deduplicate(result.items)  # L1.6 : un seul Item par item_id
    if len(result.items) < collected:
        result.warnings.append(f"{collected - len(result.items)} doublon(s) retiré(s)")
    return result
