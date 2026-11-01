"""Cloisonnement (principe P1 du plan) : le générateur d'évaluation n'est jamais lu.

Le test lit seulement nos propres fichiers (trainset et lot 2), jamais `packages/collect/synth/`.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAINSET = ROOT / "packages" / "collect" / "trainset"
ANALYSE = ROOT / "packages" / "analyse"


def _imports(folder: Path) -> dict[Path, set[str]]:
    out = {}
    for path in folder.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names |= {f"{node.module}.{a.name}" for a in node.names}
        out[path] = names
    return out


def _offenders(folder: Path, prefix: str) -> list[str]:
    return [
        f"{path.relative_to(ROOT)} → {name}"
        for path, names in _imports(folder).items()
        for name in sorted(names)
        if name == prefix or name.startswith(prefix + ".")
    ]


def test_ni_trainset_ni_le_lot_2_n_importent_le_generateur_d_evaluation():
    assert _offenders(TRAINSET, "packages.collect.synth") == []
    assert _offenders(ANALYSE, "packages.collect.synth") == []


def test_trainset_ne_depend_pas_du_lot_2():
    assert _offenders(TRAINSET, "packages.analyse") == []
