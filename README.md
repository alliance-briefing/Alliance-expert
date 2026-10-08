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

## Voir un brief (démo)
1. `docker compose -f infra/docker-compose.yml up --build`, puis ouvrir http://localhost:8000.
2. Se connecter avec `u_demo_001` (connexion de démo, active si `AUTH_MODE=demo` dans `infra/.env`).
3. Choisir un jour entre le 05/10/2026 et le 16/11/2026, puis « Générer le brief ».

Par l'API (en-tête `X-Demo-User`) :
```bash
curl -X POST localhost:8000/briefs -H "X-Demo-User: u_demo_001" -H "Content-Type: application/json" -d '{"date": "2026-10-07"}'
curl localhost:8000/briefs/2026-10-07 -H "X-Demo-User: u_demo_001"
curl localhost:8000/items/mail:dm1-msg-000003 -H "X-Demo-User: u_demo_001"
```

| Endpoint | Rôle |
|---|---|
| `POST /briefs` | génère le brief du jour de l'utilisateur connecté |
| `GET /briefs/{date}` | brief déjà généré (cache) |
| `GET /items/{item_id}` | métadonnées et lien source d'un item |
| `GET /me`, `GET /health` | utilisateur connecté, santé |

Brancher d'autres données (lot 1) : écrire `<user>/<AAAA-MM-JJ>/items.jsonl` dans un dossier
et le désigner par `DATA_DIR` (voir `infra/.env.example`).

## Contribuer
Voir [CONTRIBUTING.md](CONTRIBUTING.md) (règles Git du §4 du guide).
