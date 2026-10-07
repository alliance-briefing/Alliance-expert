# alliance-briefing

Monorepo du projet Alliance Experts — brief quotidien.

| Dossier | Propriétaire |
|---|---|
| `contracts/` | 3 référents |
| `packages/collect/` | Lot 1 |
| `packages/analyse/` | Lot 2 |
| `apps/`, `eval/`, `infra/`, `docs/` | Lot 3 |

## Prérequis
Git ≥ 2.40, Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop, `pre-commit` (`uv tool install pre-commit`).

## Démarrage (< 15 min)
```bash
git clone https://github.com/alliance-briefing/Alliance-expert.git
cd Alliance-expert
git config pull.rebase true
git config rebase.autoStash true

uv sync --all-packages            # installe Python 3.12 et les dépendances
pre-commit install                # ruff + gitleaks à chaque commit

uv run pytest                     # tests

cp infra/.env.example infra/.env  # valeurs factices, à adapter
docker compose -f infra/docker-compose.yml up --build
# http://localhost:8000/health  → {"status":"ok"}
# http://localhost:8000/docs    → documentation OpenAPI
```

## Contribuer
Voir [CONTRIBUTING.md](CONTRIBUTING.md) (règles Git du §4 du guide).
