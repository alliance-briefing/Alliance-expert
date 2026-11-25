"""Interface LLM et client factice pour les tests (L2.5 — stub)."""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Interface minimale : un appel texte → texte JSON."""

    name: str

    def complete(self, prompt: str, *, temperature: float = 0) -> str:
        """Envoie `prompt` au modèle, renvoie la réponse brute (JSON attendu)."""
        ...


class FakeLLMClient:
    """Client factice pour les tests : renvoie un résumé construit à partir du texte d'entrée.

    Pas de réseau, pas de tokens. Le résultat est déterministe.
    """

    name = "fake"

    def complete(self, prompt: str, *, temperature: float = 0) -> str:
        # Extrait le texte de l'entrée du prompt pour construire une réponse réaliste.
        text = ""
        action = None
        for line in prompt.splitlines():
            if line.startswith("TEXTE_ACTUEL:"):
                text = line.removeprefix("TEXTE_ACTUEL:").strip()
            if line.startswith("SOURCE_1:"):
                # Utilise le premier item source pour l'action recommandée.
                source = line.removeprefix("SOURCE_1:").strip()
                if source:
                    action = f"Traiter : {source[:60]}"
        summary = text if text else "Élément à traiter."
        return json.dumps(
            {"text": summary, "recommended_action": action},
            ensure_ascii=False,
        )
