"""Évaluation du brief (L3.6) : génère le rapport daté du jeu choisi.

uv run python -m eval                         # jeu de dev → eval/reports/<date>_<commit>_dev.md
uv run python -m eval --set hidden            # jeu caché (eval/hidden/, hors dépôt)
uv run python -m eval --stdout                # affiche le rapport sans l'écrire

Code de sortie 0 même si une cible n'est pas atteinte : c'est un rapport, pas un test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contracts.demo_data import DEMO_DIR
from eval.report import code_version, evaluate, latency_summary, render
from packages.analyse import generate_brief

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
SETS = {
    "dev": (EVAL / "annotations" / "dev", DEMO_DIR),
    "hidden": (EVAL / "hidden" / "annotations", EVAL / "hidden"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description=__doc__)
    parser.add_argument("--set", choices=sorted(SETS), default="dev")
    parser.add_argument("--seed", type=int, default=0, help="graine du bootstrap")
    parser.add_argument("--out", type=Path, default=EVAL / "reports")
    parser.add_argument("--stdout", action="store_true", help="affiche sans écrire de fichier")
    args = parser.parse_args(argv)

    annotations, items = SETS[args.set]
    if not any(annotations.glob("*.json")):
        raise SystemExit(f"aucune annotation dans {annotations}")
    evaluation = evaluate(annotations, items, generate_brief, args.set, args.seed)
    version = code_version(ROOT)
    report = render(evaluation, version)
    if args.stdout:
        sys.stdout.write(report)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{version[1]}_{version[0]}_{args.set}.md"
        path.write_text(report, encoding="utf-8")
        print(f"rapport écrit : {path}")
    print(latency_summary(evaluation), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
