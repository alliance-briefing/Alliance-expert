"""Configuration commune des tests.

Le brief note les affaires avec le score appris (MiniLM, 0,22 Go à télécharger). Les tests
gardent par défaut le score par règles : rapides, sans réseau, et inchangés pour ceux qui
ne portent pas sur le score. Les tests du score appris passent un faux modèle explicitement
(`packages/analyse/tests/test_learned_in_brief.py`).
"""

import os

os.environ.setdefault("ALLIANCE_SCORE", "regles")
