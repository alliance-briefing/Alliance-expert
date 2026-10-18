# packages/analyse — lot 2 (Analyse)

Transforme les `Item` d'un utilisateur (contrat A) en `Brief` (contrat B).

## État actuel : brief à base de règles (L2.0), par affaires (L2.1)

```python
from contracts.demo_data import load_items
from packages.analyse import generate_brief

brief = generate_brief(load_items("2026-10-07"), "u_demo_001", "2026-10-07")
```

| Section | Règle | Une entrée = | Ordre |
|---|---|---|---|
| `actions` | mails non lus, 5 affaires au plus | les mails non lus d'une affaire | mail non lu le plus récent d'abord |
| `meetings` | événements qui occupent au moins une partie du jour | une réunion (jamais fusionnée : une heure par entrée) | heure de début |
| `overdue` | tâches en retard à l'heure de la collecte (`Item.is_overdue`) | les tâches en retard d'une affaire | échéance la plus ancienne d'abord |

Si l'affaire contient d'autres éléments, l'entrée les annonce dans une deuxième raison
(« dossier 2026-3025 : aussi 1 réunion, 1 tâche ») et les cite dans `source_item_ids` :
aucune phrase sans source.

- Isolation : seuls les items dont `owner_user_id` est l'utilisateur demandé sont lus ; les autres sont comptés dans `warnings`.
- Déterminisme : même entrée, même brief, quel que soit l'ordre des items (hors `generated_at` et `stats.latency_ms`).
- Les textes ne reprennent que les titres et les dates, jamais le contenu des mails.
- Heure de Paris (`clock.py`) : toutes les dates sont lues en Europe/Paris quel que soit le décalage
  reçu (un item en UTC venant de Graph est donc bien placé), `generated_at` compris. Exception :
  un événement de minuit à minuit dans son propre décalage est une journée entière, dont les dates
  se lisent telles quelles (le contrat A n'a pas de champ « toute la journée »).
- Scores fixes par section (retards 70, actions 60, réunions 50) en attendant le score expliqué (L2.2).
- `generator.llm = "rules"` : aucun LLM, aucun coût.

## Agrégation en affaires (L2.1, niveaux 1 et 2)

```python
from packages.analyse.aggregate.cases import group_into_cases

cases = group_into_cases(items)  # list[Case] : case_id, items, links
```

Deux items sont dans la même affaire s'ils partagent un lien ; les liens sont transitifs.

| Niveau | Lien | Exemple de `links` |
|---|---|---|
| 1 | même `thread_id` | `fil:conv:dm1-conv-000024` |
| 1 | même `correlation_id` (contrat A 1.1 : invitation reçue par mail + événement, rempli par le lot 1) | `réunion:meet:point-flash:2026-10-21T07:00Z` |
| 2 | même référence dans le **titre** | `dossier:2026-3025` |

Le type d'une référence dépend du mot qui précède le numéro (`REFERENCE_KINDS`) :
dossier, mission, rapport, devis → `dossier` ; facture → `facture` ; commande → `commande` ;
tout autre mot ne crée aucun lien. Une facture et une commande au même numéro restent donc séparées.

- Titre seulement : un extrait cite souvent d'autres dossiers et fusionnerait des affaires sans rapport.
- Isolation : deux utilisateurs ne partagent jamais une affaire.
- Déterminisme : `case_id = "case:" + plus petit item_id` ; affaires triées par `case_id`,
  items par date de création.
- Le niveau 3 (sujet proche, embeddings) viendra si la mesure du lot 3 montre un rappel
  insuffisant.

## Commandes

```bash
uv run pytest packages/analyse --cov=packages/analyse   # tests et couverture
uv run ruff check . && uv run ruff format --check .     # style
```

## Arborescence

```
aggregate/rules.py   règles de sélection (fonctions pures)
aggregate/cases.py   regroupement des items en affaires (L2.1)
brief.py             generate_brief : assemble le Brief
clock.py             heure de Paris et détection des journées entières
tests/               fabrique d'Item synthétiques + tests
```
