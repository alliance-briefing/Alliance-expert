# Modèle de menaces

Périmètre : collecte (lot 1), stockage des Items, journal d'audit, et ce que l'API (lot 3) et
l'analyse (lot 2) en font. Rédigé par le lot 1 (L1.7), à relire et approuver par le lot 3.
Mis à jour le 17/11/2026. ✅ en place · ⏳ issue ouverte.

## Actifs

| Actif | Où | Sensibilité |
|---|---|---|
| Jetons OAuth (Microsoft, Google) | aujourd'hui : variable d'environnement ; demain : stockage chiffré (#87) | critique : accès à toute la boîte |
| Items (titres, extraits, participants) | `DATA_DIR/<user>/<jour>/items.jsonl` | personnelle, confidentielle |
| Corps complets des messages | jamais stockés (`content_text = null`) | — |
| Journal d'audit | `DATA_DIR/<user>/audit.jsonl` | identifiants seulement |
| Briefs générés | cache en mémoire de l'API | personnelle |
| Index vectoriel et cache d'embeddings (lot 2) | en mémoire, par utilisateur | personnelle (dérivée) |
| Secrets de l'application OAuth | `.env` local, jamais versionné | critique |

## Menaces et mesures

| # | Menace | Mesures | État |
|---|---|---|---|
| M1 | **Un utilisateur lit les données d'un autre** (API, fichiers, index) | `owner_user_id` vérifié à la sortie de chaque connecteur (`runner`) et à l'écriture (`store`) ; un dossier par utilisateur, nom contrôlé (pas de `../`) ; l'API refuse un item d'autrui (403) ; index du lot 2 par utilisateur. Test d'intrusion : `packages/collect/tests/test_securite.py` | ✅ |
| M2 | **Fuite d'un jeton** (journaux, sortie, erreurs, Git) | les erreurs Graph ne citent que le code HTTP ; le runner ne note que le type d'erreur ; gitleaks en pre-commit et en CI ; jeton jamais écrit dans un fichier | ✅ partiel · ⏳ grep automatisé des journaux en CI (#87) |
| M3 | **Droits excessifs** | scopes en lecture seule uniquement (`Mail.Read`, `Calendars.Read`, `Tasks.Read`, `Files.Read`), un par connecteur (`required_scopes`) | ✅ · ⏳ liste justifiée (#91) |
| M4 | **Vol des jetons au repos** | chiffrement Fernet, clé dans l'environnement ; révocation en une commande | ⏳ #87 |
| M5 | **Conservation excessive** | `content_text` jamais stocké ; journées supprimées après 30 jours, audit après 90 (`collect.toml`, `admin purge`) | ✅ |
| M6 | **Droit à l'effacement non respecté** | `admin delete-user --yes` efface journées et audit ; le cache de l'API se vide au redémarrage ; l'index du lot 2 a `IndexStore.delete_user` | ✅ lot 1 · ⏳ appel automatique lot 2 / lot 3 quand ils persisteront |
| M7 | **Données personnelles dans le journal d'audit** | contrat C : identifiants et compteurs seulement, tout « @ » refusé ; erreur = type seulement (test avec une source qui cite un jeton et une adresse) | ✅ (#85) |
| M8 | **Injection de prompt** via un mail (« ignore les consignes… ») | le contenu est une donnée, jamais une instruction (lot 2) ; garde déterministe des claims (L2.4) ; le lot 1 ne transmet qu'un extrait de 500 caractères | ✅ lot 2 |
| M9 | **Contenu piégé** (HTML, script, caractères de contrôle) | HTML converti en texte, scripts et styles retirés ; caractères de contrôle refusés par le contrat ; liens en https seulement ; l'interface échappe à l'affichage | ✅ |
| M10 | **Objet fournisseur invalide transmis au lot 2** | couche unique de normalisation : validation par le contrat, rejet journalisé sans contenu | ✅ |
| M11 | **Données réelles dans le dépôt** | données fictives seulement (domaines `.test`, `.example`) ; aucun compte client (Copilot Studio exclu) | ✅ |
| M12 | **Source indisponible ou bloquante** (429, 5xx, réseau) | réessais 1-2-4-8 s avec `Retry-After`, 5 essais ; une source en panne n'arrête pas les autres et laisse une trace d'audit | ✅ |

## Risques acceptés

- L'API garde les briefs en mémoire : une suppression demande son redémarrage, tant qu'aucune
  route de suppression n'existe côté lot 3.
- Le journal d'audit est en ajout seulement, sauf la rétention qui retire les lignes de plus de
  90 jours (réécriture d'un bloc).
- Pas de chiffrement des Items au repos : protégés par les droits du système de fichiers du
  serveur, durée limitée à 30 jours.

## Commandes

```bash
uv run python -m packages.collect.admin delete-user --user <id>          # aperçu
uv run python -m packages.collect.admin delete-user --user <id> --yes    # suppression
uv run python -m packages.collect.admin purge                            # rétention
```
