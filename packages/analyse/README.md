# packages/analyse — lot 2 (Analyse)

Transforme les `Item` d'un utilisateur (contrat A) en `Brief` (contrat B).

## État actuel : brief à base de règles (L2.0)

```python
from contracts.demo_data import load_items
from packages.analyse import generate_brief

brief = generate_brief(load_items("2026-10-07"), "u_demo_001", "2026-10-07")
```

| Section | Règle | Ordre |
|---|---|---|
| `actions` | 5 mails non lus les plus récents | du plus récent au plus ancien |
| `meetings` | événements qui occupent au moins une partie du jour | heure de début |
| `overdue` | tâches en retard à l'heure de la collecte (`Item.is_overdue`) | échéance la plus ancienne d'abord |

- Isolation : seuls les items dont `owner_user_id` est l'utilisateur demandé sont lus ; les autres sont comptés dans `warnings`.
- Déterminisme : même entrée, même brief, quel que soit l'ordre des items (hors `generated_at` et `stats.latency_ms`).
- Les textes ne reprennent que les titres et les dates, jamais le contenu des mails.
- Scores fixes par section (retards 70, actions 60, réunions 50) en attendant le score expliqué (L2.2).
- `generator.llm = "rules"` : aucun LLM, aucun coût.

## Commandes

```bash
uv run pytest packages/analyse --cov=packages/analyse   # tests et couverture
uv run ruff check . && uv run ruff format --check .     # style
```

## Arborescence

```
aggregate/rules.py   règles de sélection (fonctions pures)
brief.py             generate_brief : assemble le Brief
tests/               fabrique d'Item synthétiques + tests
```
