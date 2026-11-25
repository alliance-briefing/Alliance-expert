"""Alerte de régression (L3.6) : compare les métriques de deux versions du code.

uv run python -m eval.compare base.json pr.json            # tableau Markdown, code 0
uv run python -m eval.compare base.json pr.json --strict   # code 1 s'il y a une régression

Les deux fichiers viennent de `python -m eval --json`, calculés avec le même évaluateur et
les mêmes annotations : un écart vient du code évalué, pas du hasard (tout est déterministe).
Régression = une métrique qui se dégrade de plus de 3 points (0,03). Une comparaison entre
jeux ou journées différents n'est pas valable : elle est signalée, sans alerte. Un
changement de générateur est noté, mais l'alerte reste : c'est là qu'une régression arrive.
Sur GitHub Actions, chaque alerte est aussi écrite en `::warning::` (visible dans la PR).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

THRESHOLD = 0.03  # 3 points (guide L3.6)

# clé → (libellé, True si plus haut vaut mieux)
METRICS = {
    "recall": ("Rappel@7", True),
    "precision": ("Précision@7", True),
    "ndcg": ("NDCG@5", True),
    "claims_supported": ("Claims soutenus", True),
    "texts_supported": ("Textes d'entrée soutenus", True),
    "injections_neutralized": ("Injections neutralisées", True),
    "days_with_top_fault": ("Journées avec une faute dans le top 3", False),
    "briefs_with_unsupported_claim": ("Briefs avec un claim non soutenu", False),
}


@dataclass(frozen=True)
class Row:
    key: str
    base: float | None
    head: float | None

    @property
    def label(self) -> str:
        return METRICS[self.key][0]

    @property
    def loss(self) -> float | None:
        """Dégradation (positive = pire), dans le sens de la métrique."""
        if self.base is None or self.head is None:
            return None
        higher_is_better = METRICS[self.key][1]
        return self.base - self.head if higher_is_better else self.head - self.base

    @property
    def regression(self) -> bool:
        return self.loss is not None and self.loss > THRESHOLD + 1e-9


def invalid_reasons(base: dict, head: dict) -> list[str]:
    """Pourquoi la comparaison n'est pas valable (vide = valable)."""
    reasons = []
    if base.get("set") != head.get("set"):
        reasons.append(f"jeux différents : {base.get('set')} / {head.get('set')}")
    if base.get("days") != head.get("days"):
        reasons.append("journées différentes")
    if head.get("errors"):
        reasons.append(f"brief non produit sur la PR : {', '.join(head['errors'])}")
    return reasons


def generator_note(base: dict, head: dict) -> str | None:
    """Le générateur a changé : on compare quand même (c'est là qu'une régression arrive),
    mais on le dit, car un modèle non téléchargé retombe sur les règles."""
    before, after = base.get("generators") or ["aucun"], head.get("generators") or ["aucun"]
    if before == after:
        return None
    return f"le générateur change : {', '.join(before)} → {', '.join(after)}"


def compare(base: dict, head: dict) -> list[Row]:
    return [Row(key, base["metrics"].get(key), head["metrics"].get(key)) for key in METRICS]


def _value(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}".replace(".", ",")


def _delta(row: Row) -> str:
    if row.base is None or row.head is None:
        return "—"
    points = round((row.head - row.base) * 100)
    return f"{points:+d} pt" if points else "="


def render(base: dict, head: dict, rows: list[Row], reasons: list[str]) -> str:
    regressions = [r for r in rows if r.regression]
    if reasons:
        verdict = "⚠️ Comparaison non valable : " + " ; ".join(reasons) + "."
    elif regressions:
        names = ", ".join(r.label for r in regressions)
        verdict = f"❌ Régression de plus de 3 points : {names}."
    else:
        verdict = "✅ Aucune métrique ne perd plus de 3 points."
    note = generator_note(base, head)
    out = [
        "## Évaluation du brief : base contre PR",
        "",
        verdict,
        "",
        *([f"ℹ️ Note : {note} (vérifier que c'est voulu).", ""] if note else []),
        (
            f"Base `{base.get('code')}`, PR `{head.get('code')}` ; jeu « {head.get('set')} », "
            f"{len(head.get('days', []))} journée(s) ; générateur : "
            f"{', '.join(head.get('generators') or ['aucun'])}."
        ),
        "",
        "| Métrique | Base | PR | Écart | |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        flag = "❌" if row.regression and not reasons else ""
        out.append(
            f"| {row.label} | {_value(row.base)} | {_value(row.head)} | {_delta(row)} | {flag} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.compare", description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--strict", action="store_true", help="code 1 en cas de régression")
    parser.add_argument(
        "--github", action="store_true", help="::warning:: et résumé du job (GitHub Actions)"
    )
    args = parser.parse_args(argv)

    base = json.loads(args.base.read_text(encoding="utf-8"))
    head = json.loads(args.head.read_text(encoding="utf-8"))
    rows, reasons = compare(base, head), invalid_reasons(base, head)
    table = render(base, head, rows, reasons)
    sys.stdout.write(table)
    regressions = [] if reasons else [r for r in rows if r.regression]
    if args.github:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:  # résumé affiché sur la page du job
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(table)
        for reason in reasons:
            print(f"::warning title=Évaluation non comparable::{reason}")
        note = generator_note(base, head)
        if note:
            print(f"::notice title=Évaluation::{note}")
        for r in regressions:
            print(
                f"::warning title=Régression de l'évaluation::{r.label} : "
                f"{_value(r.base)} → {_value(r.head)} ({_delta(r)})"
            )
    return 1 if args.strict and regressions else 0


if __name__ == "__main__":
    sys.exit(main())
