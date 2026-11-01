"""Génération du corpus d'entraînement : 40 personas × 15 jours ouvrés.

`uv run python -m packages.collect.trainset` écrit, pour chaque persona :
- `data/items/<persona>.jsonl` : les `Item` du contrat A (ce que verrait le lot 2) ;
- `data/scenario/<persona>.jsonl` : les faits et étiquettes (jamais montrés au modèle) ;
puis `data/manifest.json` (empreintes et distribution) et un petit échantillon versionné.

Le corpus n'est pas versionné : il se régénère à l'identique. `--verifier` régénère en
mémoire et compare aux empreintes du manifeste versionné.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from contracts.models import Item
from packages.collect.trainset.cases import Sim
from packages.collect.trainset.facts import Fact, label
from packages.collect.trainset.world import Persona, all_personas
from packages.collect.trainset.writer import Writer

DATA = Path(__file__).with_name("data")
MANIFEST = DATA / "manifest.json"
SAMPLE_PERSONAS = ("logistique-1", "hotel-1")
SAMPLE_DAYS = 2


def scenario_row(fact: Fact, persona: Persona, templates: list[str]) -> dict:
    labels = label(fact)
    ends = fact.meta.get("end") or fact.meta.get("ends_at")
    return {
        "item_id": fact.item_id,
        "persona": persona.key,
        "metier": persona.metier.key,
        "split": persona.split,
        "source": fact.source,
        "case_id": fact.case_id,
        "case_type": fact.case_type,
        "action": labels["action"],
        "urgence": labels["urgence"],
        "alerte": labels["alerte"],
        "visible_from": fact.at.isoformat(),
        "resolved_at": fact.resolved_at.isoformat() if fact.resolved_at else None,
        "ends_at": ends.isoformat() if ends else None,
        "author_kind": fact.author.kind,
        "form": fact.form if fact.source == "mail" else None,
        "deadline": fact.deadline.isoformat() if fact.deadline else None,
        "to_manager": fact.to_manager,
        "sent": fact.sent,
        "trap": fact.trap,
        "links": fact.links,
        "templates": templates,
    }


def build_persona(persona: Persona, seed: int = 0) -> tuple[list[Item], list[dict]]:
    writer = Writer(persona, seed)
    items, rows = [], []
    for fact in Sim(persona, seed).run():
        item, templates = writer.write(fact)
        items.append(item)
        rows.append(scenario_row(fact, persona, templates))
    return items, rows


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def render(persona: Persona, seed: int = 0) -> tuple[str, str]:
    items, rows = build_persona(persona, seed)
    return _jsonl([i.model_dump(mode="json") for i in items]), _jsonl(rows)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def distribution(rows: list[dict]) -> dict:
    by_source: dict[str, Counter] = {}
    for r in rows:
        by_source.setdefault(r["source"], Counter())[r["action"]] += 1
    return {
        "elements": len(rows),
        "actions_par_source": {s: dict(sorted(c.items())) for s, c in sorted(by_source.items())},
        "urgence": dict(sorted(Counter(r["urgence"] for r in rows).items())),
        "alertes": sum(r["alerte"] for r in rows),
        "pieges": dict(sorted(Counter(r["trap"] for r in rows if r["trap"]).items())),
        "affaires": dict(sorted(Counter(r["case_type"] for r in rows).items())),
    }


def build_all(out: Path = DATA, seed: int = 0) -> dict:
    (out / "items").mkdir(parents=True, exist_ok=True)
    (out / "scenario").mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": seed, "personas": {}}
    all_rows: list[dict] = []
    for persona in all_personas(seed):
        items_text, rows_text = render(persona, seed)
        for folder, text in (("items", items_text), ("scenario", rows_text)):
            (out / folder / f"{persona.key}.jsonl").write_text(text, encoding="utf-8", newline="\n")
        rows = [json.loads(x) for x in rows_text.splitlines()]
        all_rows.extend(rows)
        manifest["personas"][persona.key] = {
            "split": persona.split,
            "elements": len(rows),
            "items_sha256": _sha(items_text),
            "scenario_sha256": _sha(rows_text),
        }
    manifest["distribution"] = {
        split: distribution([r for r in all_rows if r["split"] == split])
        for split in ("train", "tgen")
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _sample(out, seed)
    return manifest


def _sample(out: Path, seed: int) -> None:
    """Petit échantillon versionné : 2 jours de 2 personas (1 d'entraînement, 1 de T-gen)."""
    (out / "sample").mkdir(parents=True, exist_ok=True)
    for persona in all_personas(seed):
        if persona.key not in SAMPLE_PERSONAS:
            continue
        items, rows = build_persona(persona, seed)
        days = sorted({i.updated_at.date() for i in items})[:SAMPLE_DAYS]
        keep = [k for k, i in enumerate(items) if i.updated_at.date() in days]
        (out / "sample" / f"{persona.key}.items.jsonl").write_text(
            _jsonl([items[k].model_dump(mode="json") for k in keep]), encoding="utf-8", newline="\n"
        )
        (out / "sample" / f"{persona.key}.scenario.jsonl").write_text(
            _jsonl([rows[k] for k in keep]), encoding="utf-8", newline="\n"
        )


def verify(manifest_path: Path = MANIFEST, keys: list[str] | None = None) -> list[str]:
    """Personas dont la régénération ne donne pas les empreintes du manifeste."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for persona in all_personas(manifest["seed"]):
        if keys and persona.key not in keys:
            continue
        expected = manifest["personas"][persona.key]
        items_text, rows_text = render(persona, manifest["seed"])
        if (_sha(items_text), _sha(rows_text)) != (
            expected["items_sha256"],
            expected["scenario_sha256"],
        ):
            bad.append(persona.key)
    return bad


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if "--verifier" in argv:
        bad = verify()
        print("Corpus identique au manifeste." if not bad else f"Écarts : {', '.join(bad)}")
        return 1 if bad else 0
    manifest = build_all()
    for split, d in manifest["distribution"].items():
        print(f"[{split}] {d['elements']} éléments, {d['alertes']} alertes, pièges {d['pieges']}")
        for source, actions in d["actions_par_source"].items():
            print(f"  {source:<9} {actions}")
    print(f"→ {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
