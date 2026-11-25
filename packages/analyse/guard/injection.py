"""Tentatives d'injection de prompt dans un élément (M9) : règle déterministe, sans modèle.

Un élément qui s'adresse à l'assistant (« assistant », « IA », « SYSTEM : »…) pour qu'il
contourne ses consignes (« ignore tes consignes », « oublie les règles », « mode
administrateur »…) n'est jamais noté ni cité par le brief : le guide du lot 3 (L3.5) le range
dans « ne doit pas figurer », et un brief qui le reprend est en faute. Le brief ne recopie
jamais le contenu d'un élément ; cette règle évite en plus qu'un tel élément monte en tête
parce que la compréhension le juge « alerte ».

Il faut les deux signes : un destinataire machine et une consigne de contournement. Un mail
qui parle d'« assistante » ou de « règles » sans les deux ne déclenche rien.
"""

from __future__ import annotations

import re
import unicodedata

from contracts.models import Item

_ADDRESSED = re.compile(
    r"\b(assistant(e)? (de tri|ia|virtuel)|l'?ia\b|intelligence artificielle|chatbot|"
    r"\bai\b|system\s*:|systeme\s*:|qui lit ce message|you are now)"
)
_OVERRIDE = re.compile(
    r"\b(ignore[rz]?|oublie[rz]?|forget|disregard)\b.{0,40}"
    r"\b(consignes?|instructions?|regles?|rules|ordres?)\b"
    r"|\b(mode administrateur|admin mode|developer mode|mode developpeur)\b"
)


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def is_injection(item: Item) -> bool:
    """L'élément contient-il une consigne adressée à l'assistant pour contourner ses règles ?"""
    text = _plain(" ".join(filter(None, (item.title, item.snippet, item.content_text))))
    return bool(_ADDRESSED.search(text) and _OVERRIDE.search(text))
