Règles de travail :
- Tout mon code va dans packages/analyse/ (sous-dossiers aggregate, scoring,
  summarize, guard, llm, rag, tests). Ne touche pas à packages/collect/ ni à apps/.
- Ne modifie JAMAIS contracts/ : si un champ te manque, dis-le-moi, ne le
  rajoute pas. N'invente aucun champ du contrat A ou B : lis les modèles dans
  contracts/models/ s'ils existent.
- Python 3.12, type hints sur les interfaces publiques, pydantic v2, pytest,
  ruff sans avertissement, couverture visée ≥ 80 % sur packages/analyse.
- Résultats déterministes : mêmes entrées, même sortie ; graine fixée si de l'aléatoire.
- Aucun secret dans le code, aucune donnée réelle : uniquement des données synthétiques.
- Git : une branche par issue (format l2/<type>/<slug>), commits au format
  Conventional Commits (type(portée): description), petits commits, PR ≤ 400 lignes.
- Avant d'écrire du code, propose-moi un plan et attends mon accord.
- Explique-moi chaque choix en quelques lignes : je dois pouvoir défendre
  chaque ligne à l'oral.