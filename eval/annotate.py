"""Outils d'annotation (L3.5). Ne lit jamais truth.json.

uv run python -m eval.annotate fiche 2026-10-07               # fiche lisible des items du jour
uv run python -m eval.annotate modele 2026-10-07 --annotateur guillaume
uv run python -m eval.annotate verifie                        # contrôle toutes les annotations
uv run python -m eval.annotate accord 2026-10-07              # kappa entre deux annotateurs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from contracts.demo_data import DEFAULT_USER, load_items
from contracts.models import Item
from eval.annotation import Annotation, check_against_day, cohen_kappa

ROOT = Path(__file__).parent / "annotations"
PARIS = ZoneInfo("Europe/Paris")
DEV_SET = "dev"
SOURCES = {"mail": "Mails", "calendar": "Réunions", "task": "Tâches", "file": "Fichiers"}


def _fmt(value: object) -> str:
    return value.strftime("%d/%m %H:%M") if hasattr(value, "strftime") else "—"


def _line(item: Item, collected: datetime) -> str:
    flags = []
    if item.source == "mail":
        flags.append("non lu" if item.is_read is False else "lu")
        if item.native_importance == "high":
            flags.append("IMPORTANT")
    if item.source == "calendar":
        flags.append(f"{_fmt(item.start_at)} → {_fmt(item.end_at)}")
    if item.source == "task":
        flags.append(f"échéance {_fmt(item.due_at)}")
        flags.append("terminée" if item.completed else "à faire")
        if item.is_overdue(collected):
            flags.append("EN RETARD")
    sender = next((p.name for p in item.participants if p.role in ("from", "organizer")), "")
    title = item.title.replace("|", "/") or "(sans objet)"
    snippet = item.snippet.replace("|", "/").replace("\n", " ")
    return (
        f"| `{item.item_id}` | {_fmt(item.created_at)} | {sender} | {', '.join(flags)} "
        f"| {title} | {snippet} |"
    )


def sheet(day: str, user: str = DEFAULT_USER) -> str:
    """Fiche Markdown de la journée : tous les items, par source, les plus récents d'abord."""
    items = load_items(day, user)
    collected = max(item.collected_at for item in items)
    out = [
        f"# Journée du {day} — {user}",
        (
            f"Collecte à {collected.strftime('%H:%M')} · {len(items)} items. "
            "Ne pas ouvrir truth.json : l'annotation doit rester indépendante."
        ),
    ]
    for source, label in SOURCES.items():
        chosen = sorted(
            (i for i in items if i.source == source), key=lambda i: i.created_at, reverse=True
        )
        out += [
            "",
            f"## {label} ({len(chosen)})",
            "",
            "| id | créé | de | état | titre | extrait |",
            "|---|---|---|---|---|---|",
        ]
        out += [_line(item, collected) for item in chosen]
    return "\n".join(out) + "\n"


def template(day: str, annotator: str, user: str = DEFAULT_USER) -> dict:
    example = {"rank": 1, "section": "actions", "item_ids": ["mail:..."], "why": "..."}
    return {
        "schema": "annotation/1.0",
        "user_id": user,
        "date": day,
        "annotator": annotator,
        "annotated_at": datetime.now(PARIS).date().isoformat(),
        "must_include": [example, {**example, "rank": 2}, {**example, "rank": 3}],
        "must_exclude": [],
        "notes": "",
    }


def annotation_files(root: Path = ROOT) -> list[Path]:
    return sorted(root.glob("*/*.json"))


def verify(root: Path = ROOT) -> list[str]:
    problems = []
    for path in annotation_files(root):
        try:
            annotation = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"{path.name} : {exc}")
            continue
        if path.name.split(".")[0] != annotation.date.isoformat():
            problems.append(f"{path.name} : la date du nom de fichier ne correspond pas")
        items = load_items(annotation.date.isoformat(), annotation.user_id)
        problems += [f"{path.name} : {p}" for p in check_against_day(annotation, items)]
    return problems


def agreement(day: str, root: Path = ROOT) -> float:
    paths = sorted(root.glob(f"*/{day}.*.json"))
    if len(paths) != 2:
        raise SystemExit(f"il faut exactement 2 annotations du {day}, trouvé {len(paths)}")
    a, b = (Annotation.model_validate_json(p.read_text(encoding="utf-8")) for p in paths)
    return cohen_kappa(a, b, load_items(day, a.user_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.annotate", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fiche", "modele", "accord"):
        cmd = sub.add_parser(name)
        cmd.add_argument("date")
        if name == "modele":
            cmd.add_argument("--annotateur", required=True)
    sub.add_parser("verifie")
    args = parser.parse_args(argv)

    if args.cmd == "fiche":
        path = ROOT / "fiches" / f"{args.date}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sheet(args.date), encoding="utf-8")
        print(f"fiche écrite : {path}")
    elif args.cmd == "modele":
        path = ROOT / DEV_SET / f"{args.date}.{args.annotateur}.json"
        if path.exists():
            raise SystemExit(f"{path} existe déjà : je ne l'écrase pas")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(template(args.date, args.annotateur), ensure_ascii=False, indent=2)
        path.write_text(content + "\n", encoding="utf-8")
        print(f"modèle écrit : {path}")
    elif args.cmd == "verifie":
        problems = verify()
        print("\n".join(problems) or f"OK : {len(annotation_files())} annotation(s) valide(s)")
        return 1 if problems else 0
    else:
        print(f"kappa de Cohen ({args.date}) : {agreement(args.date):.2f} (attendu ≥ 0,60)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
