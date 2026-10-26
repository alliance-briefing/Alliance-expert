"""Briefs de démonstration construits à partir des annotations de référence (L3.5).

Sert UNIQUEMENT aux démonstrations de l'interface (DEMO_BRIEFS=annotations) : on montre au
client un brief tel qu'un manager l'attend (classé par importance, avec le « pourquoi »),
en attendant que le générateur du lot 2 sache le produire.

- Lecture seule : les annotations ne sont jamais modifiées.
- Le générateur du lot 2 n'est ni appelé ni modifié : l'évaluation L3.5 reste indépendante.
- Un jour sans annotation retombe sur le générateur normal (`fallback`).
- Le brief porte `generator.llm = "annotation"` : l'interface l'affiche comme un aperçu.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alliance_api.briefs import BriefGenerator
from contracts.models import Brief, Entry, Generator, Item, Section, Stats
from contracts.models.brief import SECTION_ORDER
from eval.annotation import Annotation

PARIS = ZoneInfo("Europe/Paris")
ANNOTATIONS_DIR = Path(__file__).parents[4] / "eval" / "annotations" / "dev"
DEMO_GENERATOR = "annotation"
TOP_SCORE, STEP = 95, 8  # rang 1 → 95, rang 2 → 87… (l'ordre compte, pas la valeur)


def _label(item: Item, day: Date) -> str:
    """Texte de l'entrée : le titre de l'item principal, précédé de sa nature."""
    title = item.title or "(sans objet)"
    if item.source == "calendar" and item.start_at:
        start = item.start_at.astimezone(PARIS)
        when = "" if start.date() == day else f" le {start:%d/%m}"
        return f"Réunion{when} à {start:%H:%M} : {title}"
    if item.source == "task":
        late = item.due_at and item.due_at < item.collected_at
        return f"{'Tâche en retard' if late else 'Tâche'} : {title}"
    return {"mail": "Mail", "file": "Fichier"}.get(item.source, "Élément") + f" : {title}"


def brief_from_annotation(annotation: Annotation, items: Sequence[Item]) -> Brief:
    """Le brief que l'annotateur attend pour cette journée, au format du contrat B."""
    by_id = {item.item_id: item for item in items}
    ranked: dict[str, list[Entry]] = {kind: [] for kind in SECTION_ORDER}
    for expected in sorted(annotation.must_include, key=lambda e: e.rank):
        ids = [i for i in expected.item_ids if i in by_id]  # un item absent n'est pas cité
        if not ids:
            continue
        section = ranked[expected.section]
        ranked[expected.section].append(
            Entry(
                rank=len(section) + 1,
                priority_score=max(TOP_SCORE - (expected.rank - 1) * STEP, 1),
                priority_reasons=[expected.why],
                text=_label(by_id[ids[0]], annotation.date),
                source_item_ids=ids,
            )
        )
    cited = {i for entries in ranked.values() for e in entries for i in e.source_item_ids}
    return Brief(
        brief_id=f"b_{annotation.date.isoformat()}_{annotation.user_id}",
        owner_user_id=annotation.user_id,
        date=annotation.date,
        generated_at=datetime.now(PARIS),
        generator=Generator(llm=DEMO_GENERATOR),
        sections=[
            Section(kind=kind, entries=entries) for kind, entries in ranked.items() if entries
        ],
        stats=Stats(
            items_in=len(items),
            items_out=len(cited),
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            cost_eur=0.0,
        ),
    )


def annotated_generator(
    fallback: BriefGenerator, annotator: str, root: Path = ANNOTATIONS_DIR
) -> BriefGenerator:
    """Générateur qui sert l'annotation de `annotator` quand elle existe, sinon `fallback`."""

    def generate(
        items: Sequence[Item], user: str, date: Date | str, *, now: datetime | None = None
    ) -> Brief:
        day = Date.fromisoformat(date) if isinstance(date, str) else date
        path = root / f"{day.isoformat()}.{annotator}.json"
        if path.is_file():
            annotation = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
            if annotation.user_id == user:
                return brief_from_annotation(annotation, items)
        return fallback(items, user, date, now=now)

    return generate
