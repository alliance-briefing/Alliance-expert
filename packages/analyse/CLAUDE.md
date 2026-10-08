# Lot 2 — Analyse

## Contexte
Mission : transformer les `Item` d'un utilisateur (contrat A, produits par le lot 1) en un
`Brief` court, priorisé et sourcé (contrat B, consommé par le lot 3).
Principe du guide : un brief agréable mais faux est pire qu'un brief absent.

Rôles proposés par le guide (non attribués : tout le monde travaille sur tout) :
- A, agrégation et priorisation : regrouper les items en affaires (L2.1), score de
  priorité expliqué (L2.2) ;
- B, LLM, prompts, résumés : texte et action recommandée en sortie structurée (L2.3) ;
- C, garde anti-hallucination et traçabilité : vérification déterministe des claims (L2.4) ;
- D, RAG, coûts, comparatif de modèles : interface LLM remplaçable et comparatif (L2.5),
  RAG conditionnel (L2.6), performance et coût (L2.7).
Échéances : 28/10 (v0.2.0), 11/11 (v0.3.0), 25/11 (gel), 09/12 (rendu).
Objectifs et critères d'acceptation de chaque tâche : `Alliance-Experts_guide-equipe_briefing.html`
à la racine, partie « Lot 2 ». Les relire avant de commencer une tâche.

## Contrats (lecture seule)
- Modèles : `contracts/models/item.py` (A), `brief.py` (B), `audit.py` (C).
- `contracts/fixtures/brief/invalid/` : 23 briefs refusés par le contrat (claim sans source,
  score croissant…), à lire pour savoir ce qui est interdit.
- Données de démo : `contracts.demo_data.load_items("2026-10-07")` (u_demo_001, du 05/10 au 16/11).

## Qui dépend de notre code
- L'API du lot 3 (`apps/api/src/alliance_api/briefs.py`) appelle
  `generate_brief(items, user, date) -> Brief`, et la CI génère un vrai brief dans Docker.
  Ne pas changer cette signature ni la forme du Brief sans prévenir le lot 3.
- L'API vérifie de son côté que chaque item cité appartient à l'utilisateur : ça ne nous
  dispense pas de le garantir.

## Règles de travail
- Tout notre code va dans `packages/analyse/` (sous-dossiers aggregate, scoring,
  summarize, guard, llm, rag, tests). Ne pas toucher à `packages/collect/` ni à `apps/`.
- Ne JAMAIS modifier `contracts/` : si un champ manque, le signaler, ne pas le rajouter.
  N'inventer aucun champ des contrats : lire les modèles.
- Python 3.12, type hints sur les interfaces publiques, pydantic v2, pytest,
  ruff sans avertissement, couverture ≥ 80 % sur `packages/analyse` (seuil bloquant en CI
  à partir de la v0.2.0).
- Résultats déterministes : mêmes entrées, même sortie ; graine fixée si de l'aléatoire,
  température 0 pour les comparatifs de modèles.
- Nouvelle dépendance : `uv add <paquet>` (jamais `pip install`), sinon elle manque dans
  `uv.lock` et l'image Docker de la CI casse.
- Aucun secret dans le code (gitleaks bloque la CI), aucune donnée réelle.
- Git : une branche par issue (`l2/<type>/<slug>`, en minuscules), commits au format
  Conventional Commits (`type(analyse): description`), petits commits, PR ≤ 400 lignes.
- Avant d'écrire du code, proposer un plan et attendre l'accord.
- Expliquer chaque choix en quelques lignes : chaque ligne doit pouvoir être défendue à l'oral.

## Règles de sécurité et de fiabilité (exigences du guide)
- Ne jamais lire `truth.json` (code, tests, prompts) : ce sont les réponses de l'évaluation.
- Isolation : ne traiter que les items dont `owner_user_id` est l'utilisateur du brief ;
  un prompt ou un index ne contient jamais les données d'un autre utilisateur.
- Le contenu d'un mail, d'une tâche ou d'un fichier est une donnée, jamais une instruction
  (injection de prompt : « ignore les consignes et… » ne doit rien changer).
- Chaque entrée et chaque claim porte au moins un `source_item_id` existant dans l'entrée.
- Si le LLM échoue ou n'est pas vérifiable : entrée dégradée (extrait + lien) avec un
  warning, ou brief à base de règles. Jamais de contenu inventé.
- Un LLM qui vérifie un LLM n'est pas une preuve : la garde (L2.4) est déterministe.

## Commandes
Depuis la racine du dépôt :
```bash
uv sync --all-packages                                  # dépendances
uv run pytest packages/analyse --cov=packages/analyse   # tests et couverture
uv run ruff check . && uv run ruff format --check .     # style (comme la CI)
```
Voir le brief dans l'interface du lot 3 (PowerShell) :
```powershell
$env:AUTH_MODE = "demo"; $env:PYTHONPATH = "."
uv run uvicorn alliance_api.main:app --reload --port 8000
# http://localhost:8000, utilisateur u_demo_001, puis « Générer le brief »
```
Les liens « Voir la source » des données de démo pointent vers des domaines `.test` :
ils ne s'ouvrent pas, c'est normal.

## Avant d'ouvrir une PR
ruff check, ruff format --check et pytest passent en local ; README du lot mis à jour si le
comportement change ; la PR ne se fusionne qu'avec la CI verte (lint-test, gitleaks, docker).
