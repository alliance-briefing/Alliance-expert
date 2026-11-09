"""Trace d'une entrée du brief (L2.4) : chaque claim, son verdict, ses items et leurs preuves.

```bash
uv run python -m packages.analyse.guard.trace b_2026-10-07_u_demo_001 1
uv run python -m packages.analyse.guard.trace b_2026-10-07_u_demo_001 2 --section meetings
uv run python -m packages.analyse.guard.trace b_<jour>_<utilisateur> 1 --dossier <DATA_DIR>
```

Le brief est régénéré à partir des items de la journée : `generate_brief` est déterministe
(hors `generated_at` et latence), c'est donc le brief servi, à données et modèle égaux
(`ALLIANCE_SCORE=regles` pour le score par règles). Le verdict de chaque claim est recalculé
par la garde ; une date de l'item qui figure dans le claim est marquée « (citée) ».
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date as Date
from pathlib import Path

from contracts.demo_data import DEMO_DIR, load_items
from contracts.models import Brief, Entry, Item
from contracts.models.brief import SECTION_ORDER
from packages.analyse.brief import generate_brief
from packages.analyse.guard.check import check_claim, moments

SNIPPET_CHARS = 200


def parse_brief_id(brief_id: str) -> tuple[str, str]:
    """`b_<AAAA-MM-JJ>_<utilisateur>` → (jour, utilisateur) ; l'utilisateur peut contenir « _ »."""
    prefix, _, rest = brief_id.partition("_")
    day, _, user = rest.partition("_")
    try:
        Date.fromisoformat(day)
    except ValueError:
        day = ""
    if prefix != "b" or not day or not user:
        raise ValueError(f"identifiant de brief invalide : {brief_id!r} (attendu b_<jour>_<user>)")
    return day, user


def trace(brief: Brief, rank: int, items: Sequence[Item], section: str | None = None) -> str:
    """Le texte de trace des entrées de rang `rank` (de la section `section`, sinon de toutes)."""
    by_id = {item.item_id: item for item in items}
    blocks = [
        _entry(s.kind, entry, by_id, brief.owner_user_id)
        for s in brief.sections
        if section in (None, s.kind)
        for entry in s.entries
        if entry.rank == rank
    ]
    if not blocks:
        where = f"la section {section}" if section else brief.brief_id
        raise LookupError(f"aucune entrée de rang {rank} dans {where}")
    return "\n\n".join(blocks)


def _entry(kind: str, entry: Entry, by_id: dict[str, Item], user: str) -> str:
    lines = [f"[{kind} n°{entry.rank}] {entry.text} (score {entry.priority_score})"]
    if not entry.claims:
        lines.append("  aucun claim : entrée dégradée, seul le lien de la source fait foi")
    for claim in entry.claims:
        problem = check_claim(claim, by_id, user)
        verdict = "soutenu" if problem is None else f"REJETÉ : {problem}"
        lines += [f"  claim : {claim.text}", f"    verdict : {verdict}"]
        for item_id in claim.source_item_ids:
            lines += [f"    {line}" for line in _evidence(by_id.get(item_id), item_id, claim.text)]
    claimed = {item_id for claim in entry.claims for item_id in claim.source_item_ids}
    for item_id in entry.source_item_ids:
        if item_id not in claimed:
            item = by_id.get(item_id)
            lines.append(f"  contexte : {item_id} « {item.title if item else 'introuvable'} »")
    return "\n".join(lines)


def _evidence(item: Item | None, item_id: str, claim: str) -> list[str]:
    if item is None:
        return [f"{item_id} : introuvable dans les items de la journée"]
    lines = [f"{item.item_id} ({item.source}) « {item.title} »", f"  lien : {item.url}"]
    for name, value in moments(item).items():
        mark = " (citée)" if f"{value:%d/%m}" in claim else ""
        lines.append(f"  {name} : {value:%d/%m/%Y %H:%M}{mark}")
    if item.snippet:
        snippet = item.snippet[:SNIPPET_CHARS] + ("…" if len(item.snippet) > SNIPPET_CHARS else "")
        lines.append(f"  extrait : {snippet}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("brief_id", help="b_<AAAA-MM-JJ>_<utilisateur>")
    parser.add_argument("rank", type=int, help="rang de l'entrée (1 = la plus prioritaire)")
    parser.add_argument("--section", choices=SECTION_ORDER, help="par défaut : toutes")
    parser.add_argument("--dossier", type=Path, default=DEMO_DIR, help="DATA_DIR (défaut : démo)")
    args = parser.parse_args(argv)
    try:
        day, user = parse_brief_id(args.brief_id)
        items = load_items(day, user, args.dossier)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        print(trace(generate_brief(items, user, day), args.rank, items, args.section))
    except LookupError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
