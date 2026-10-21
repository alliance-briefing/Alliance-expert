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
| `meetings` | événements non annulés (`attributes.calendar.is_cancelled`) qui occupent au moins une partie du jour | une réunion (jamais fusionnée : une heure par entrée) | heure de début |
| `overdue` | tâches en retard à l'heure de la collecte (`Item.is_overdue`) | les tâches en retard d'une affaire | échéance la plus ancienne d'abord |

Si l'affaire contient d'autres éléments, l'entrée les annonce dans une deuxième raison
(« dossier 2026-3025 : aussi 1 réunion, 1 tâche ») et les cite dans `source_item_ids` :
aucune phrase sans source.

- Isolation : seuls les items dont `owner_user_id` est l'utilisateur demandé sont lus ; les autres sont comptés dans `warnings`.
- Déterminisme : même entrée, même brief, quel que soit l'ordre des items (hors `generated_at` et `stats.latency_ms`).
- Les textes ne reprennent que les titres et les dates, jamais le contenu des mails.
- Heure de Paris (`clock.py`) : toutes les dates sont lues en Europe/Paris quel que soit le décalage
  reçu (un item en UTC venant de Graph est donc bien placé), `generated_at` compris. Exception :
  une journée entière, dont les dates se lisent telles quelles. Elle est donnée par
  `attributes.calendar.is_all_day` (contrat A 1.1) ; sans ce bloc (items 1.0), on la déduit :
  minuit à minuit dans son propre décalage.
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
| 2 | même document : le fichier et les titres qui le citent entre « » (nom avec extension) | `document:Planning_experts_S41.xlsx` |

Le type d'une référence dépend du mot qui précède le numéro (`REFERENCE_KINDS`) :
dossier, mission, expertise, rapport, devis → `dossier` ; facture → `facture` ; commande → `commande` ;
tout autre mot ne crée aucun lien. Une facture et une commande au même numéro restent donc séparées.
Les « _ » comptent comme des espaces : `Rapport_expertise_2026-2367.docx` rejoint le dossier 2026-2367.
Un texte entre guillemets sans extension (« Expertises complexes ») est un canal de discussion :
il ne crée aucun lien.

- Titre seulement : un extrait cite souvent d'autres dossiers et fusionnerait des affaires sans rapport.
- Isolation : deux utilisateurs ne partagent jamais une affaire.
- Déterminisme : `case_id = "case:" + plus petit item_id` ; affaires triées par `case_id`,
  items par date de création.
- Le niveau 3 (sujet proche, embeddings) viendra si la mesure du lot 3 montre un rappel
  insuffisant, derrière un veto « références différentes » : l'exploration (`rag/explore.py`)
  montre que les messages-types (« Dossier 2026-5394 : RAS » / « Dossier 2026-4752 : RAS »)
  sont presque identiques pour le modèle.

## Embeddings (L2.6, en cours)

`rag/embeddings.py` : vecteurs de sens des items, base du futur niveau 3 de L2.1 et du RAG.
Pas encore utilisé par le brief.

```python
from packages.analyse.rag.embeddings import FastEmbedder, embed_items, similarities

vectors = embed_items(items, FastEmbedder())  # {item_id: vecteur normalisé}
```

- `Embedder` remplaçable : `FastEmbedder` (fastembed, `paraphrase-multilingual-MiniLM-L12-v2`,
  384 dimensions, 0,22 Go, local, sans PyTorch) ou `HashEmbedder` (faux modèle des tests).
- Texte vectorisé (`item_text`) : titre sans « RE: », « TR : »…, puis extrait.
- Cache par (modèle, propriétaire, item_id, updated_at) : un item n'est jamais recalculé.
- Dépendance optionnelle, absente de la CI et de l'image Docker :
  `uv sync --all-packages --group embeddings`. Le modèle est téléchargé au premier appel ;
  sans fastembed, le test du vrai modèle est ignoré.
- Mesure locale (CPU) : chargement 2,6 s, puis 500 textes en 1,4 s ; similarité d'une paraphrase
  française 0,71 contre 0,16 pour un texte sans rapport.

### Index par utilisateur (`rag/index.py`, étape R1)

```python
from packages.analyse.rag.embeddings import FastEmbedder
from packages.analyse.rag.index import IndexStore

store = IndexStore(FastEmbedder())
index = store.for_user("u_demo_001")
index.add(items_des_jours_passes)
hits = index.search_items(items_du_jour, date_du_brief, exclude=ids_du_jour)  # k=3, 30 jours
store.delete_user("u_demo_001")  # suppression complète
```

- Isolation : un index par utilisateur ; il refuse les items et les requêtes d'un autre.
- Pas de fuite du futur : seuls les items créés avant le jour du brief (heure de Paris), dans
  les 30 jours précédents, sont renvoyés.
- Une version plus récente d'un item remplace l'ancienne ; tri déterministe (score, puis item_id).
- En mémoire (NumPy) ; le stockage persistant sera tranché dans l'ADR RAG. Pas encore branché
  dans le brief (étapes R2-R3).

### Historique des affaires (`rag/retrieve.py`, étape R2)

`history(cases, index, jour)` → pour chaque affaire du jour, au plus 3 items passés liés
(`PastLink` : item, score, raison). Rien n'est encore affiché dans le brief (R3).

- Requête : chaque item de l'affaire interroge l'index ; un item passé garde sa meilleure similarité.
- Même référence (« même dossier 2026-3025 ») : lien certain, accepté quel que soit le score.
- Sinon similarité ≥ seuil, sauf veto si l'affaire et l'item citent des références toutes
  différentes. Seuil provisoire 0,85 (prudent), à régler avec `explore --historique` (R2b).
- Liens certains d'abord, puis score, puis item_id ; items du jour et de l'affaire exclus.

### Exploration (`rag/explore.py`)

Ce que trouverait un niveau 3 de L2.1 : distribution des similarités (paires déjà reliées par les
niveaux 1-2 contre les autres) et paires les plus proches non reliées, à juger à la main.

```bash
uv run python -m packages.analyse.rag.explore 2026-10-07                 # titre + extrait
uv run python -m packages.analyse.rag.explore 2026-10-07 --texte titre --top 40
uv run python -m packages.analyse.rag.explore 2026-10-07 --modele hash   # faux modèle, instantané
uv run --with model2vec python -m packages.analyse.rag.explore 2026-10-07 --modele model2vec
uv run python -m packages.analyse.rag.explore <jour> --dossier <DATA_DIR> --utilisateur <id>
```

Historique des affaires (R2b) : l'index reçoit les N journées précédentes, puis chaque affaire
du jour affiche ses items passés retenus et écartés, avec la raison (`retrieve.verdict`).

```bash
uv run python -m packages.analyse.rag.explore 2026-10-09 --historique --jours-historique 5 --seuil 0.85
```

Résultats (démo, MiniLM-L12, 5 jours d'historique, 09/10, 15/10 et 21/10, jugés à la main) :
les liens de même référence sont justes ; les rattachements « sujet proche » sont faux dans
toutes les tranches de score jugées (0,75 à 1,00) : noms de fichiers générés
(`Compte_rendu_20261007_01.docx` ← `Compte_rendu_20261006_52.docx`, photos `IMG_…`),
occurrences d'une même réunion récurrente, messages-types (« Déjeuner – Maëlle » ←
« Déjeuner – Jérôme »). Les scores les plus hauts sont les plus trompeurs. Réserve : données de
démo générées à partir de modèles de phrases. Le seuil par défaut (0,85) est en attente de
décision ; piste : similarité désactivée par défaut, ou encadrée par des garde-fous
(participant commun, fichiers et séries exclus).

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
rag/embeddings.py    vecteurs de sens des items (L2.6)
rag/explore.py       outil d'exploration des embeddings sur une journée
rag/index.py         index RAG par utilisateur (L2.6, R1)
rag/retrieve.py      historique des affaires du jour (L2.6, R2)
tests/               fabrique d'Item synthétiques + tests
```
