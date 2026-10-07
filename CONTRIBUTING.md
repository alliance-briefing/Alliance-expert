# Contribuer

Résumé du §4 du guide d'équipe. En cas de doute, le guide fait foi.

## Branches
- `main` est protégée : pas de push direct, PR obligatoire, 1 approbation, CI verte, branche à jour, squash merge.
- Une issue par tâche (`L1.3`, `L2.4`…). Pas de code sans issue.
- Une branche par issue : `<lot>/<type>/<slug>`, ex. `l1/feat/mail-connector`. Types : `feat fix docs test refactor chore`.
- Une branche vit **2 jours ouvrés maximum**. Rebase sur `main` au moins une fois par jour.
- Après le merge, la branche est supprimée (automatique).

## Pull requests
- ≤ 400 lignes modifiées (hors fichiers générés et fixtures).
- Gabarit `.github/PULL_REQUEST_TEMPLATE.md` rempli.
- 1 approbation d'une personne du même lot ; 2 si la PR touche `contracts/` (un référent de chaque lot). Personne n'approuve sa propre PR.
- Réponse à une demande de revue sous 24 h ouvrées.
- `contracts/` se modifie dans une PR séparée (étiquette `contract-change`).

## Commits
Conventional Commits avec le paquet en portée :
```
feat(collect): ajoute le connecteur calendrier Graph
fix(analyse): corrige la dédup des fils
docs(adr): ADR-0004 choix du modèle
chore(ci): active gitleaks
```
Un commit = une intention. Pas de « wip », « fix », « update ».

## Secrets et données
- Aucun secret dans Git. `.env` local, non versionné ; modèle dans `infra/.env.example`.
- `gitleaks` tourne en pre-commit et en CI.
- Secret poussé : 1) le révoquer, 2) prévenir le référent sécurité (lot 1), 3) ensuite seulement nettoyer l'historique.
- Aucune donnée réelle de personne.

## Interdit
- `git push --force` sur `main` ou une branche partagée (sur ta branche après rebase : `--force-with-lease`).
- `git add -A` / `git add .` sans relire `git status`.
- Committer des fichiers > 5 Mo, des modèles, des bases locales, `.venv`, `node_modules`, `__pycache__`.

## Commandes
```bash
git switch main && git pull
git switch -c l1/feat/mail-connector
git add -p
git commit -m "feat(collect): liste les mails des dernières 24 h"
git fetch origin && git rebase origin/main
git push -u origin l1/feat/mail-connector
gh pr create --fill --base main
```

## Avant de pousser
```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Décisions
Toute décision coûteuse à défaire → ADR dans `docs/adr/NNNN-titre.md` (gabarit `docs/adr/0000-gabarit.md`), par PR.
