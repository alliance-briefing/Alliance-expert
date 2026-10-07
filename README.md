# alliance-briefing

Monorepo du projet Alliance Experts — brief quotidien.

| Dossier | Propriétaire |
|---|---|
| `contracts/` | 3 référents |
| `packages/collect/` | Lot 1 |
| `packages/analyse/` | Lot 2 |
| `apps/`, `eval/`, `infra/`, `docs/` | Lot 3 |

## Démarrage
```bash
uv sync --dev
pre-commit install
cp infra/.env.example infra/.env
```

Règles Git : voir §4 du guide d'équipe.
