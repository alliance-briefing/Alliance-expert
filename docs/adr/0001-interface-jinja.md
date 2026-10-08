# ADR-0001 : interface rendue côté serveur (FastAPI + Jinja, HTMX si besoin)
Date : 2026-10-07 · Statut : proposé
Décideurs : référents lot 1, 2, 3

## Contexte (3 lignes)
L3.3 demande une interface de consultation du brief, maintenable par une équipe de 4 qui
connaît Python mais peu JavaScript. Contraintes : clavier seul, WCAG AA, 375 px, aucun
HTML d'e-mail injecté tel quel, chaque texte affiché avec son lien source.

## Options examinées (2 à 4, avec un chiffre ou un test pour chacune)
1. **FastAPI + Jinja (HTMX ajouté seulement si une interaction l'exige)** : 1 langage,
   0 dépendance npm, 1 conteneur ; échappement HTML automatique (test `<script>` du 14/10).
2. **SPA Vite + React** : 2e langage et chaîne npm (~200 paquets), 2e conteneur ou build
   à servir, CORS et jetons côté navigateur ; échappement automatique aussi (JSX).
3. **Streamlit** : rapide à prototyper, mais contrôle faible du HTML (accessibilité,
   lien source par texte) et pas d'authentification par session adaptée.

## Décision
Option 1. Les pages sont servies par l'API (`alliance_api.ui`), avec les mêmes stores,
le même contrôle d'utilisateur et le même journal d'audit que les endpoints JSON.

## Conséquences (ce que ça coûte, comment on revient en arrière)
Interactions riches (retour par entrée sans rechargement) : ajouter HTMX (un script, pas
de build). Retour arrière : l'API JSON reste complète, une SPA peut la consommer sans
toucher au lot 2 ; seules les pages `ui.py` et `templates/` seraient abandonnées.
