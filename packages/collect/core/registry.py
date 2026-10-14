"""Registre : trouve tout seul les connecteurs présents dans packages/collect/connectors/.

Chaque sous-dossier de connectors/ qui contient une sous-classe de Connector est une source.
Elles sont toutes actives, sauf celles listées dans `disabled` de collect.toml.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import tomllib
from pathlib import Path

from packages.collect import connectors as connectors_package
from packages.collect.core.connector import Connector

CONFIG_PATH = Path(__file__).resolve().parents[1] / "collect.toml"


def discover() -> dict[str, type[Connector]]:
    """Toutes les classes de connecteur trouvées, rangées par nom."""
    found: dict[str, type[Connector]] = {}
    for module_info in pkgutil.iter_modules(connectors_package.__path__):
        module = importlib.import_module(f"{connectors_package.__name__}.{module_info.name}")
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, Connector) and not inspect.isabstract(obj):
                found[obj.name] = obj
    return found


def disabled_names(config_path: Path = CONFIG_PATH) -> set[str]:
    """Noms des connecteurs désactivés dans le fichier de configuration."""
    if not config_path.exists():
        return set()
    config = tomllib.loads(config_path.read_text("utf-8"))
    return set(config.get("connectors", {}).get("disabled", []))


def enabled_connectors(config_path: Path = CONFIG_PATH) -> list[Connector]:
    """Une instance de chaque connecteur actif, dans l'ordre alphabétique des noms."""
    off = disabled_names(config_path)
    return [cls() for name, cls in sorted(discover().items()) if name not in off]
