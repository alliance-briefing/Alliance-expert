"""Ligne de commande du générateur.

Exemples :
    uv run synth                                   # 30 journées, graine 42, dans contracts/fixtures/demo
    uv run synth --seed 7 --days 5 --out /tmp/essai
    uv run synth --users u_demo_001 u_demo_002     # deux managers (tests d'isolation)
    uv run synth --seed 2027 --out eval/hidden     # jeu de test caché (lot 3 uniquement)
    uv run synth --explique 2026-10-21             # résumé lisible d'une journée déjà générée
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path

from collect.synth.generator import DEFAULT_START, Params, generate
from collect.synth.people import PROFILES

SECTIONS = {"actions": "À traiter", "meetings": "Réunion", "overdue": "En retard", "files": "Fichier"}


def explain(out: Path, user_id: str, day: str) -> str:
    """Résumé en français d'une journée : ce que le manager reçoit et ce que le brief doit dire."""
    folder = out / user_id / day
    items = [json.loads(line) for line in (folder / "items.jsonl").read_text("utf-8").splitlines()]
    truth = json.loads((folder / "truth.json").read_text("utf-8"))
    by_id = {it["item_id"]: it for it in items}
    counts = Counter(it["source"] for it in items)
    lines = [
        f"Journée du {day} — type « {truth['day_type']} » — collecte à {truth['collected_at'][11:16]}",
        f"Collecté : {counts['mail']} mails, {counts['calendar']} événements, {counts['task']} tâches, "
        f"{counts['file']} fichiers ({len(truth['noise_item_ids'])} éléments de bruit)",
        "",
        "Ce que le brief DOIT contenir, dans cet ordre :",
    ]
    for entry in truth["expected_brief"]:
        lines.append(
            f"  {entry['rank']}. [{SECTIONS[entry['section']]}] {entry['attendu']}  (priorité {entry['salience']})"
        )
        lines.append(f"     pourquoi : {', '.join(entry['priority_reasons'])}")
        for source_id in entry["source_item_ids"]:
            lines.append(f"     source : {by_id[source_id]['title'] or '(sans objet)'}  <{source_id}>")
    for note in truth["notes"]:
        lines.append(f"\nÀ signaler ({note['kind']}) : {note.get('attendu', note.get('explication', ''))}")
    if truth["prompt_injections"]:
        lines.append(f"\nPièges d'injection à ignorer : {len(truth['prompt_injections'])}")
    traps = {k: len(v) for k, v in truth["traps"].items() if k not in ("malformed_html", "recurring")}
    if traps:
        lines.append("Autres pièges présents : " + ", ".join(f"{k} ×{n}" for k, n in sorted(traps.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth", description="Génère des journées de manager factices (L1.0).")
    parser.add_argument("--seed", type=int, default=42, help="graine (même graine = mêmes fichiers)")
    parser.add_argument("--days", type=int, default=30, help="nombre de journées ouvrées exportées")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="premier jour (AAAA-MM-JJ)")
    parser.add_argument("--users", nargs="+", default=["u_demo_001"], choices=sorted(PROFILES))
    parser.add_argument("--out", type=Path, default=Path("contracts/fixtures/demo"))
    parser.add_argument(
        "--with-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="écrit aussi les corps HTML bruts des mails, piège « HTML mal formé » (désactiver : --no-with-raw)",
    )
    parser.add_argument("--explique", metavar="AAAA-MM-JJ", help="affiche le résumé d'une journée déjà générée")
    args = parser.parse_args(argv)

    if args.explique:
        print(explain(args.out, args.users[0], args.explique))
        return 0

    started = time.perf_counter()
    manifest = generate(args.out, Params(args.seed, args.days, args.start, tuple(args.users), args.with_raw))
    totals = manifest["totals"]
    print(f"{len(manifest['days'])} journées écrites dans {args.out} en {time.perf_counter() - started:.1f} s")
    print(
        f"  mails {totals['mail']} · événements {totals['calendar']} · tâches {totals['task']} · fichiers {totals['file']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
