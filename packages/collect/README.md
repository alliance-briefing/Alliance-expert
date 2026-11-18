# `packages/collect` — collecte (lot 1)

Va chercher les mails, réunions, tâches et fichiers d'un utilisateur et les rend au format
**contrat A** (`contracts/models/item.py`) : une liste d'`Item` validés, lue par le lot 2.

```
connecteurs (un dossier chacun) ──► normalisation (L1.6) ──► items.jsonl ──► API / lot 2
        │                                                        │
  jetons OAuth chiffrés (L1.2)                    journal d'audit (L1.8), rétention (L1.7)
```

## Lancer

Tout se lance depuis la racine du dépôt, après `uv sync --all-packages`.

```powershell
# Données de démonstration (contracts/fixtures/demo) : rien à configurer
uv run python -m packages.collect.cli --user u_demo_001 --day 2026-10-05

# Persona Graph fictive (Hélène MARCHAND) : réponses Graph enregistrées dans tests/fixtures/graph
uv run python -m packages.collect.cli --user u_demo_graph --day 2026-10-21

# Une seule source, même si elle est désactivée dans collect.toml
uv run python -m packages.collect.cli --user u_demo_graph --day 2026-10-21 --source mail_graph

# Écrire la journée pour l'API au lieu de l'afficher : <DATA_DIR>/<user>/<jour>/items.jsonl
uv run python -m packages.collect.cli --user u_demo_graph --day 2026-10-21 --write --data-dir data
```

La collecte d'un jour `J` a lieu à **8h30, heure de Paris** ; chaque connecteur choisit sa
propre fenêtre à partir de cet instant (voir le tableau ci-dessous). Sans `--write`, les Items
sortent sur la sortie standard (une ligne JSON par Item) et le résumé sur la sortie d'erreur.
Un connecteur qui échoue ne bloque pas les autres : il devient un `WARNING` (L1.1).

## Les sources

| Connecteur | Source | Fenêtre et règles | Scope | Variable pour le vrai compte |
|---|---|---|---|---|
| `mail_graph` | Outlook (L1.3) | reçus dans les 24 h + adressés directement et sans réponse depuis 7 jours ; 500 au plus | `Mail.Read` | `MAIL_SOURCE=graph` |
| `calendar_graph` | Calendrier Outlook (L1.4) | 48 prochaines heures + événement en cours ; récurrences dépliées | `Calendars.Read` | `CALENDAR_SOURCE=graph` |
| `tasks_graph` | Microsoft To Do (L1.4) | en retard (50 au plus) + dues dans les 24 h ; sans échéance = jamais collectée | `Tasks.Read` | `TASKS_SOURCE=graph` |
| `files_graph` | OneDrive (L1.5) | créés ou modifiés dans les 48 h ; 100 au plus ; extrait pour txt, md, csv, docx (< 2 Mo), métadonnées seules sinon | `Files.Read` | `FILES_SOURCE=graph` |
| `files_graph` | OneDrive (L1.5) | créés ou modifiés dans les 48 h ; 100 au plus ; métadonnées seules | `Files.Read` | `FILES_SOURCE=graph` |
| `demo` | fixtures `contracts/fixtures/demo` (L1.0) | journées générées, toutes sources | aucun | — |
| `notes_demo` | notes Markdown factices (source n°5, L1.1) | notes écrites dans la fenêtre | aucun | — |

Par défaut, chaque connecteur Graph lit des **réponses enregistrées** (`tests/fixtures/graph/`),
servies au seul utilisateur `u_demo_graph` : aucun réseau, aucun compte nécessaire. Les
variables `MAIL_GRAPH_DIR`, `CALENDAR_GRAPH_DIR`, `TASKS_GRAPH_DIR` et `FILES_GRAPH_DIR`
pointent vers un autre jeu au même format (par exemple le jeu complet
`alliance-graph-synth/data` pour les mails).

## Configurer

### `collect.toml`

```toml
[connectors]
disabled = []            # noms des connecteurs à couper, ex. ["notes_demo"]

[oauth.microsoft]
client_id = "…"          # application « Alliance Briefing - demo » (MS_CLIENT_ID la remplace)
tenant = "common"

[retention]              # facultatif ; valeurs par défaut du guide (L1.7)
keep_content_text = false   # corps complet jamais écrit sur disque
snippet_days = 30           # journées d'Items conservées 30 jours
audit_days = 90             # journal d'audit conservé 90 jours
```

### Variables d'environnement

| Variable | Rôle |
|---|---|
| `DATA_DIR` | dossier des journées (`<user>/<jour>/items.jsonl`) et du journal (`<user>/audit.jsonl`) |
| `TOKEN_KEY` | clé Fernet de chiffrement des jetons (sinon : fichier écrit par `auth keygen --save`) |
| `TOKENS_DIR` | dossier des jetons chiffrés (défaut : `~/.local/share/alliance-briefing/tokens`) |
| `MS_CLIENT_ID` | remplace le `client_id` de `collect.toml` |
| `GRAPH_ACCESS_TOKEN` | jeton saisi à la main pour un essai ponctuel ; jamais à versionner |
| `MAIL_SOURCE`, `CALENDAR_SOURCE`, `TASKS_SOURCE`, `FILES_SOURCE` | `graph` = vrai compte ; sinon fixtures |

Les modèles de valeurs sont dans `infra/.env.example`. Le fichier `.env` n'est jamais versionné.

### Brancher un vrai compte Microsoft

Enregistrement de l'application chez Microsoft : `docs/oauth-microsoft.md`. Puis :

```powershell
uv run python -m packages.collect.auth keygen --save            # une fois : clé de chiffrement
uv run python -m packages.collect.auth login  --user u_claire   # ouvre le navigateur
uv run python -m packages.collect.auth status --user u_claire   # connecté ? jusqu'à quand ?
$env:MAIL_SOURCE = "graph"
uv run python -m packages.collect.cli --user u_claire --day 2026-11-17
uv run python -m packages.collect.auth logout --user u_claire   # efface les jetons
```

Seuls des scopes en **lecture seule** sont demandés ; le jeton d'accès est renouvelé tout
seul, et un accès révoqué donne l'erreur explicite `reauth_required`.

## Données d'un utilisateur

```powershell
uv run python -m packages.collect.admin delete-user --user u_claire          # aperçu
uv run python -m packages.collect.admin delete-user --user u_claire --yes    # suppression
uv run python -m packages.collect.admin purge                                # rétention
```

Tout ce qui concerne un utilisateur est rangé sous `DATA_DIR/<user>/` (isolation L1.7) :
le supprimer efface ses journées et son journal d'audit, et `delete-user` efface aussi
ses jetons OAuth (`TOKENS_DIR`). Menaces et mesures :
`docs/threat-model.md`.

## Ajouter une source

Le registre (`core/registry.py`) trouve tout seul chaque sous-dossier de `connectors/` qui
contient une sous-classe de `Connector`. **Ajouter une source ne modifie aucun fichier
existant** : on crée un dossier, c'est tout.

1. Créer `connectors/ma_source/__init__.py` qui importe la classe :

   ```python
   from packages.collect.connectors.ma_source.connector import MaSourceConnector

   __all__ = ["MaSourceConnector"]
   ```

2. Écrire `connectors/ma_source/connector.py` :

   ```python
   from collections.abc import Iterator

   from contracts.models import Item
   from packages.collect.core.connector import Connector, Health, Window


   class MaSourceConnector(Connector):
       # nom unique : --source, collect.toml, audit, warnings
       name = "ma_source"
       version = "0.1.0"

       def required_scopes(self) -> list[str]:
           # autorisations en lecture seule ; vide pour une source locale
           return []

       def collect(self, user: str, window: Window) -> Iterator[Item]:
           # un Item par élément de `user` ; window.end = heure de collecte
           ...

       def health(self) -> Health:
           return Health(True, "source joignable")
   ```

3. Ajouter ses tests (dans le dossier du connecteur ou dans `tests/`).

Règles à respecter, vérifiées par les tests des connecteurs existants :

- **Isolation** : ne renvoyer que des Items de `user`, avec `acl.visible_to = [user]`. Des
  données de test ne sont servies qu'à leur utilisateur de démonstration.
- **Contrat** : passer les objets bruts par `core.normalize.normalize_all` : un objet invalide
  est rejeté seul et journalisé sans son contenu, au lieu de faire échouer la collecte.
- **Heures** : convertir avec `core.paris_time.to_paris` (décalage `+01:00` / `+02:00` explicite).
- **Microsoft Graph** : passer par `core.graph_http.get_all` (pagination, réessais 429/5xx,
  `Retry-After`) et `core.auth.get_access_token` ; le jeton ne doit jamais apparaître dans un
  journal ni une erreur.
- **Lecture seule** : déclarer chaque scope dans `required_scopes()` et le justifier dans
  `docs/oauth-microsoft.md`.

Exemple complet et minimal : `connectors/notes_demo/` (preuve L1.1, sans réseau ni jeton).
Modèle d'un connecteur Graph : `connectors/tasks_graph/`.

## Tests

```sh
uv run pytest packages/collect --cov=packages/collect --cov-report=term-missing
```

La CI exige au moins **80 %** de couverture sur le code du lot 1, mesurée hors tests et hors
`synth/` et `trainset/` (code du lot 2) : **97 %** au 18/11. Les tests ne font aucun appel
réseau : les réponses Graph viennent des fixtures ou d'un faux serveur.

### Pannes couvertes (`tests/test_robustesse.py`)

Pour chacun des 4 connecteurs Graph (mail, calendrier, tâches, fichiers) :

| Panne | Comportement | Réessais |
|---|---|---|
| Réseau coupé | warning du runner, aucun Item de cette source | 5 essais (1, 2, 4, 8 s) |
| 401 jeton refusé | warning, aucun Item | aucun |
| Connexion retirée ou expirée | warning `AuthError` (`reauth_required`) : relancer `auth login` | aucun |
| 403 accès interdit, 404 introuvable | warning, aucun Item | aucun |
| 429 trop de requêtes | réessai en respectant `Retry-After` ; warning si 5 échecs | 5 essais |
| 500, 503 panne Microsoft | réessai ; warning si 5 échecs | 5 essais |
| JSON tronqué | warning, aucun Item | aucun |
| Page suivante manquante (404) | warning, **aucun Item partiel** de cette source | aucun |
| Un objet hors contrat A | cet objet seul est rejeté, les autres sont gardés | — |
| Fichier inaccessible ou supprimé | pas d'Item pour ce fichier, les autres sont gardés | — |

Dans tous les cas, **les autres sources de l'utilisateur continuent** et le jeton n'apparaît
jamais dans les journaux ni dans les warnings.

### Collecte réelle planifiée

`.github/workflows/demo-graph.yml` collecte chaque matin de semaine le compte de démo fictif
(`u_claire`) contre le vrai Microsoft Graph, hors CI et sans bloquer les PR. Il ne fait rien tant
que les secrets `DEMO_TOKEN_KEY` et `DEMO_TOKENS_U_CLAIRE` ne sont pas définis (voir le fichier).

## Performance

Objectif L1.9 : 500 mails, 20 événements, 30 tâches, 100 fichiers en **moins de 30 s** (p95
sur 10 exécutions). Mesure : `uv run python -m packages.collect.bench` (Microsoft simulé,
tout le reste réel : pagination, JSON, sélection, HTML, contrat A, runner).

| Mesure (18/11/2026, Mac M-series) | p50 | p95 |
|---|---|---|
| Volume du guide, sans délai réseau (10 exécutions) | 0,08 s | 0,09 s |
| Volume du guide, 1 s par requête Graph (`--latency 1.0`, 3 exécutions) | 25,3 s | 25,3 s |
| Compte de démo réel (40 Items, vrai Microsoft) | ~3 s par source, 13 à 19 s au total avec le démarrage | |

Le temps est dominé par Microsoft (0,4 à 2 s par requête mesurés). Les contenus de fichiers
sont lus 16 à la fois ; les sources passent l'une après l'autre. Avec 1 s par requête, le
budget est tenu avec peu de marge : si besoin, lancer les sources en parallèle dans le runner.
