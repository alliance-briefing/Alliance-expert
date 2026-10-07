# contracts/ — contrats partagés entre les 3 lots

Contenu conforme au guide (§2.5) : modèles pydantic v2 (source de vérité), JSON Schema exporté, fixtures, CHANGELOG. **Aucun code métier d'un lot** : ce dossier se teste et s'utilise seul.

```
contracts/
├── models/            contrat A (Item) et C (AuditEvent) exécutables
├── schemas/           JSON Schema exportés (pour les outils hors Python)
├── fixtures/
│   ├── demo/          30 journées de la manager fictive u_demo_001 (générées par le lot 1)
│   ├── valid/         3 exemples d'Item valides (mail, calendrier, tâche)
│   └── invalid/       21 cas rejetés (url absente, date sans fuseau, champ inconnu…)
├── demo_data.py       lecture des journées en 1 ligne
├── validate_fixtures.py   validation en CI
├── tests/             tests du dossier (aucune dépendance aux lots)
└── CHANGELOG.md
```

## Pour le lot 2 : utiliser les journées sans rien d'autre

```python
from contracts.demo_data import available_days, load_items

for day in available_days():  # 30 jours, du 2026-10-05 au 2026-11-16
    items = load_items(day)  # list[Item] du contrat A, déjà validés
    brief = generate_brief(items, "u_demo_001", day)
```

Chaque journée = ce que le lot 1 a collecté à 08h30 : 60 à 250 mails, 4 à 10 événements, 3 à 12 tâches, 5 à 30 fichiers, avec des pièges (injections de prompt, fil de 16 messages, mail sans objet, changement d'heure…).

`truth.json` (ce que le brief devrait contenir) est **réservé à l'évaluation du lot 3** : ne pas s'en servir pour régler les règles du lot 2 (guide L3.5).

## Commandes

```bash
uv run python -m contracts.validate_fixtures     # les fixtures valident le schéma
uv run python -m contracts.export_schemas        # régénère les JSON Schema après une modification des modèles
uv run pytest contracts                          # tests du dossier
```

## Modifier un contrat

Toute modification passe par une PR avec l'étiquette `contract-change` et l'approbation d'un référent de chaque lot. Les fixtures sont mises à jour dans la même PR, et le CHANGELOG aussi.
