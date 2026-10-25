"""Échantillon de mails à étiqueter : un par modèle de phrase et par type d'expéditeur.

Le texte de David vient de modèles de phrases : 2 900 mails reçus hors bruit, mais 181
modèles une fois masqués les nombres, les jours, les mois et la partie variable. Étiqueter dix variantes du même mail n'apprend rien,
et les mettre des deux côtés de la mesure la fausserait : on prend un mail par modèle, et
`template` sert ensuite à séparer entraînement et test.

`uv run python -m packages.analyse.scoring.ml.sample <alliance-graph-synth>/data` écrit
`scoring/annotations/intent_sample.jsonl` (versionné : données synthétiques).
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from contracts.models import Item
from packages.analyse.scoring.ml.graph_convert import GraphDataset, load_graph_dataset
from packages.analyse.scoring.ml.noise import noise_reason
from packages.analyse.scoring.ml.persona import GRAPH_HELENE, SENDER_KINDS, Persona

ANNOTATIONS = Path(__file__).resolve().parent.parent / "annotations"
SAMPLE_PATH = ANNOTATIONS / "intent_sample.jsonl"
TEXT_MAX_CHARS = 1200
_PREFIX = re.compile(r"^\s*((re|tr|fw|fwd)\s?:\s*)+", re.IGNORECASE)
_CALENDAR = re.compile(
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|janvier|février|mars|avril|mai"
    r"|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
    re.IGNORECASE,
)
_VARIABLE_TAIL = re.compile(r"\s+(—|–|→).*$")  # « Planning — Refonte du CRM », « Lyon → Lille »


def template(title: str) -> str:
    """Le modèle de phrase d'un objet : sans « RE : », sans la partie variable (projet,
    trajet), nombres, jours et mois masqués. « RE : Préparation du comité du samedi 7
    novembre » → « Préparation du comité du # # # »."""
    key = _VARIABLE_TAIL.sub("", _PREFIX.sub("", title))
    key = re.sub(r"\d+", "#", _CALENDAR.sub("#", key))
    return re.sub(r"\s+", " ", key).strip() or "(sans objet)"


def _sender(item: Item) -> tuple[str, str]:
    who = next(p for p in item.participants if p.role == "from")
    return who.email, who.name


def build_sample(
    dataset: GraphDataset, persona: Persona = GRAPH_HELENE, seed: int = 0
) -> list[dict]:
    """Mails reçus hors bruit, un tiré au hasard par (type d'expéditeur, modèle)."""
    pools: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in sorted(dataset.items, key=lambda i: i.item_id):
        extras = dataset.extras.get(item.item_id)
        if extras is None or extras.is_sent or noise_reason(item):
            continue
        pools[persona.sender_kind(_sender(item)[0])][template(item.title)].append(item)
    rng = random.Random(seed)
    sample = []
    for kind in SENDER_KINDS:
        for key in sorted(pools[kind]):
            item = rng.choice(pools[kind][key])
            email, name = _sender(item)
            sample.append(
                {
                    "item_id": item.item_id,
                    "template": key,
                    "sender_kind": kind,
                    "sender": name or email,
                    "title": item.title,
                    "text": (item.content_text or item.snippet)[:TEXT_MAX_CHARS],
                }
            )
    return sample


def load_sample(path: Path = SAMPLE_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_sample(sample: list[dict], path: Path = SAMPLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(row, ensure_ascii=False) for row in sample)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-2])
        return 2
    sample = build_sample(load_graph_dataset(argv[0]))
    write_sample(sample)
    counts: dict[str, int] = defaultdict(int)
    for row in sample:
        counts[row["sender_kind"]] += 1
    print(f"{len(sample)} mails → {SAMPLE_PATH} {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
