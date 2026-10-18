"""Critère d'acceptation L1.0 : le même générateur avec la même graine produit les mêmes fichiers.

Tout est généré dans un dossier temporaire : contracts/fixtures/demo/ n'est jamais modifié
(le lot 3 cite ses item_id dans ses annotations).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from contracts.validate_fixtures import DEFAULT_DIR, validate_dir
from packages.collect.synth.generator import Params, generate


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_same_seed_gives_the_same_files(tmp_path: Path) -> None:
    params = Params(seed=7, days=3, start=date(2026, 10, 5), users=("u_demo_001",))
    generate(tmp_path / "a", params)
    generate(tmp_path / "b", params)
    assert _files(tmp_path / "a") == _files(tmp_path / "b")
    assert validate_dir(tmp_path / "a").ok


def test_other_seed_gives_other_files(tmp_path: Path) -> None:
    base = {"days": 2, "start": date(2026, 10, 5), "users": ("u_demo_001",)}
    generate(tmp_path / "a", Params(seed=7, **base))
    generate(tmp_path / "b", Params(seed=8, **base))
    assert _files(tmp_path / "a") != _files(tmp_path / "b")


def test_committed_fixtures_are_rebuilt_identically(tmp_path: Path) -> None:
    """Graine 42 (paramètres du manifeste) : on retrouve contracts/fixtures/demo/ octet pour octet."""
    manifest = json.loads((DEFAULT_DIR / "manifest.json").read_text("utf-8"))
    p = manifest["params"]
    params = Params(
        p["seed"], p["days"], date.fromisoformat(p["start"]), tuple(p["users"]), p["with_raw"]
    )
    assert params.seed == 42
    generate(tmp_path, params)
    assert _files(tmp_path) == _files(DEFAULT_DIR), (
        "les fixtures du dépôt ne correspondent plus au générateur"
    )
