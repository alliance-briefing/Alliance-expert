"""Rétention et suppression des données collectées (guide L1.7).

Politique par défaut du guide, réglable dans `collect.toml` (section `[retention]`) :
- `content_text` (corps complet) **non conservé** : retiré à l'écriture ;
- journées d'Items (titres, extraits) : 30 jours, puis supprimées ;
- journal d'audit : 90 jours, puis les lignes plus anciennes sont retirées.

`purge` applique la politique à tout `DATA_DIR` ; `delete_user` efface en une fois tout ce
que le lot 1 garde d'un utilisateur (journées, journal d'audit). Les deux sont lancés par
`python -m packages.collect.admin`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from contracts.models import Item
from packages.collect.core.paris_time import paris_midnight
from packages.collect.core.registry import CONFIG_PATH
from packages.collect.core.store import user_folder

AUDIT_FILE = "audit.jsonl"
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class RetentionPolicy:
    keep_content_text: bool = False
    snippet_days: int = 30
    audit_days: int = 90


def load_policy(config_path: Path = CONFIG_PATH) -> RetentionPolicy:
    """La politique de `collect.toml` ; valeurs du guide pour ce qui n'y est pas."""
    if not config_path.exists():
        return RetentionPolicy()
    section = tomllib.loads(config_path.read_text("utf-8")).get("retention", {})
    return RetentionPolicy(**section)


def without_content(items: Iterable[Item], policy: RetentionPolicy) -> list[Item]:
    """Les Items tels qu'ils peuvent être stockés : sans corps complet si la politique l'exige."""
    if policy.keep_content_text:
        return list(items)
    return [item.model_copy(update={"content_text": None}) for item in items]


@dataclass
class PurgeReport:
    days_removed: list[Path] = field(default_factory=list)
    audit_lines_removed: int = 0


def purge(root: Path | str, today: date, policy: RetentionPolicy) -> PurgeReport:
    """Supprime ce qui dépasse la durée de conservation, pour chaque utilisateur de `root`."""
    report = PurgeReport()
    oldest_day = today - timedelta(days=policy.snippet_days)
    oldest_event = paris_midnight(today - timedelta(days=policy.audit_days))
    root = Path(root)
    if not root.exists():
        return report
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for day in sorted(p for p in folder.iterdir() if p.is_dir() and _DAY.fullmatch(p.name)):
            if date.fromisoformat(day.name) < oldest_day:
                shutil.rmtree(day)
                report.days_removed.append(day)
        report.audit_lines_removed += _trim_audit(folder / AUDIT_FILE, oldest_event)
    return report


def _trim_audit(path: Path, oldest: datetime) -> int:
    """Retire les événements antérieurs à `oldest`. Seule exception à l'« ajout seulement » :
    la rétention, qui réécrit le fichier d'un bloc (fichier temporaire puis remplacement)."""
    if not path.exists():
        return 0
    lines = path.read_text("utf-8").splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if line.strip() and datetime.fromisoformat(json.loads(line)["ts"]) >= oldest
    ]
    if len(kept) == len(lines):
        return 0
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".audit-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(kept)
    os.replace(tmp, path)
    return len(lines) - len(kept)


def delete_user(root: Path | str, user: str) -> list[Path]:
    """Efface tout ce que le lot 1 garde de `user` ; renvoie les fichiers supprimés."""
    folder = user_folder(root, user)  # refuse « ../ » et les noms dangereux
    if not folder.exists():
        return []
    removed = sorted(p for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder)
    return removed
