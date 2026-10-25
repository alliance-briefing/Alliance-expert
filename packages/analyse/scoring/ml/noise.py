"""Filtre du bruit certain, avant le score appris (pipeline A : règles → ML).

Une seule règle : un marqueur d'envoi en masse (lien de désabonnement) dans l'objet ou le
corps. Un envoi en masse doit proposer le désabonnement ; l'adresse de l'expéditeur
(« noreply ») ne dit rien de l'importance : un rappel RH « noreply » demande une action.
Le mot « newsletter » n'est pas un marqueur : un collègue peut parler de la newsletter interne.
Tout le reste (portail RH, tickets, banque, sécurité, catalogues…) est noté par le modèle.

Contrôle : `uv run python -m packages.analyse.scoring.ml.noise <alliance-graph-synth>/data`
"""

from __future__ import annotations

import random
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass

from contracts.models import Item

REASON = "lettre d'information (lien de désabonnement)"


def _normalize(text: str) -> str:
    """Minuscules sans accents : « Désabonner » → « desabonner »."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


@dataclass(frozen=True)
class NoiseConfig:
    """Marqueurs d'envoi en masse ; casse et accents ignorés."""

    bulk_markers: frozenset[str] = frozenset(
        {"unsubscribe", "désabonner", "désabonnement", "désinscrire", "désinscription"}
    )


def noise_reason(item: Item, config: NoiseConfig | None = None) -> str | None:
    """La raison si l'item est du bruit certain, sinon None."""
    markers = {_normalize(m) for m in (config or NoiseConfig()).bulk_markers}
    text = _normalize(f"{item.title} {item.content_text or item.snippet}")
    return REASON if any(marker in text for marker in markers) else None


def _sender(item: Item) -> str:
    return next((p.email for p in item.participants if p.role == "from"), "?")


def report(items: list[Item], config: NoiseConfig | None = None, sample: int = 10) -> str:
    """Écartés par expéditeur, puis un échantillon tiré au hasard (graine fixée) à relire."""
    noisy = [item for item in items if noise_reason(item, config)]
    lines = [f"écartés : {len(noisy)} sur {len(items)}"]
    lines += [f"  {n:>5}  {sender}" for sender, n in Counter(map(_sender, noisy)).most_common()]
    lines.append("échantillon :")
    for item in random.Random(0).sample(noisy, min(sample, len(noisy))):
        lines.append(f"  - {item.title[:70]} | {item.snippet[:90]}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    from packages.analyse.scoring.ml.graph_convert import load_graph_dataset

    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1])
        return 2
    dataset = load_graph_dataset(argv[0])
    received = [
        item
        for item in dataset.items
        if item.source == "mail" and not dataset.extras[item.item_id].is_sent
    ]
    print(report(received))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
