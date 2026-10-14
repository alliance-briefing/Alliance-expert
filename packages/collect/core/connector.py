"""Interface que tous les connecteurs du lot 1 respectent (guide L1.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from contracts.models import Item


@dataclass(frozen=True)
class Window:
    """Période de collecte : de `start` à `end`, avec fuseau horaire obligatoire."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Window : les dates doivent avoir un fuseau horaire")
        if self.end < self.start:
            raise ValueError("Window : end est avant start")


@dataclass(frozen=True)
class Health:
    """État d'un connecteur : ok ou non, avec une explication courte."""

    ok: bool
    detail: str = ""


class Connector(ABC):
    """Une source de données. Ajouter une source = créer un dossier avec une sous-classe."""

    name: str
    version: str

    @abstractmethod
    def required_scopes(self) -> list[str]:
        """Autorisations en lecture seule nécessaires (vide pour une source locale)."""

    @abstractmethod
    def collect(self, user: str, window: Window) -> Iterator[Item]:
        """Les Items de `user` pour la période `window`, au format du contrat A."""

    @abstractmethod
    def health(self) -> Health:
        """Vérifie que la source répond, sans rien collecter."""
