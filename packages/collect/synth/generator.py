"""Orchestration : monde → réparation des volumes → journées exportées → fichiers.

Arborescence produite :
    <out>/manifest.json                      paramètres, compteurs, empreintes sha256
    <out>/<user_id>/<AAAA-MM-JJ>/items.jsonl  un Item (contrat A) par ligne
    <out>/<user_id>/<AAAA-MM-JJ>/truth.json   vérité terrain (pré-annotation lot 3)
    <out>/<user_id>/<AAAA-MM-JJ>/raw_mail_html.jsonl  (option --with-raw) corps HTML bruts

Aucune date « du jour », aucun identifiant aléatoire système : deux exécutions avec
la même graine produisent des fichiers identiques octet pour octet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from collect.synth.people import PROFILES
from collect.synth.repair import repair
from collect.synth.truth import build_truth
from collect.synth.window import CONNECTOR_VERSION, snapshot, to_items
from collect.synth.world import World

GENERATOR_VERSION = CONNECTOR_VERSION
DEFAULT_START = date(2026, 10, 5)


@dataclass(frozen=True)
class Params:
    seed: int = 42
    days: int = 30
    start: date = DEFAULT_START
    users: tuple[str, ...] = ("u_demo_001",)
    with_raw: bool = False


def _dump_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build_world(user_id: str, params: Params) -> World:
    world = World(PROFILES[user_id], params.start, params.days, params.seed).build()
    world.repair_rounds = repair(world)
    return world


def generate(out_dir: Path, params: Params | None = None) -> dict:
    params = params or Params()
    out_dir = Path(out_dir)
    checksums: dict[str, str] = {}
    per_day: list[dict] = []
    for user_id in params.users:
        if user_id not in PROFILES:
            raise ValueError(f"profil inconnu : {user_id} (disponibles : {', '.join(PROFILES)})")
        world = build_world(user_id, params)
        for day in world.export_days:
            snap = snapshot(world, day)
            items, index = to_items(world, snap)
            truth = build_truth(world, snap, items, index)
            folder = out_dir / user_id / day.isoformat()
            lines = "".join(
                json.dumps(it.model_dump(mode="json"), ensure_ascii=False) + "\n" for it in items
            )
            rel = f"{user_id}/{day.isoformat()}"
            checksums[f"{rel}/items.jsonl"] = _write(folder / "items.jsonl", lines)
            checksums[f"{rel}/truth.json"] = _write(folder / "truth.json", _dump_json(truth))
            if params.with_raw:
                raw = "".join(
                    json.dumps(
                        {"item_id": f"mail:{m.native_id}", "html": m.html}, ensure_ascii=False
                    )
                    + "\n"
                    for m in snap.mails
                )
                checksums[f"{rel}/raw_mail_html.jsonl"] = _write(
                    folder / "raw_mail_html.jsonl", raw
                )
            per_day.append(
                {
                    "user_id": user_id,
                    "date": day.isoformat(),
                    "day_type": truth["day_type"],
                    **truth["counts"],
                    "expected_brief_entries": len(truth["expected_brief"]),
                }
            )

    manifest = {
        "generator": {"name": "collect.synth", "version": GENERATOR_VERSION},
        "contract": {"item": "1.0", "truth": "truth/1.0"},
        "params": {
            "seed": params.seed,
            "days": params.days,
            "start": params.start.isoformat(),
            "users": list(params.users),
            "with_raw": params.with_raw,
        },
        "timezone": "Europe/Paris",
        "days": per_day,
        "totals": {k: sum(d[k] for d in per_day) for k in ("mail", "calendar", "task", "file")},
        "sha256": dict(sorted(checksums.items())),
    }
    _write(out_dir / "manifest.json", _dump_json(manifest))
    return manifest
