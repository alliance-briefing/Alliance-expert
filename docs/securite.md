# Sécurité (L3.9)

Vue d'ensemble pour l'application entière. Détails : [threat-model.md](threat-model.md)
(collecte et données, rédigé par le lot 1, menaces M1 à M12) et [auth.md](auth.md)
(connexion, sessions, droits, L3.4).

## Principes

- **Données synthétiques uniquement** dans le dépôt, les tests et les prompts (domaines `.test`,
  `.example`) ; aucun compte client.
- **Lecture seule** chez le fournisseur : un scope par connecteur (`Mail.Read`, `Calendars.Read`,
  `Tasks.Read`, `Files.Read`).
- **Isolation par utilisateur** partout : un dossier par utilisateur au nom contrôlé (pas de
  `../`), `owner_user_id` vérifié à la collecte, à l'écriture et par l'API (403 sur la ressource
  d'un autre).
- **Le contenu est une donnée, jamais une instruction** : le brief ne reprend que des titres et
  des dates, jamais l'extrait d'un mail ; chaque fait cite son élément source.
- **Journal sans données personnelles** : contrat C, identifiants et compteurs, tout « @ » refusé.

## Jetons et secrets

| Secret | Où | Protection |
|---|---|---|
| Jetons OAuth Microsoft (accès aux boîtes) | `TOKENS_DIR`, fichiers chiffrés | Fernet, fichier `0600`, clé `TOKEN_KEY` hors du dépôt ; effacés par `delete-user` |
| Secret client OpenID Connect | `infra/.env` (facultatif, PKCE suffit) | jamais versionné ; gitleaks en pre-commit et en CI |
| Jetons d'identité OpenID Connect | non conservés | vérifiés (signature, `aud`, `iss`, `exp`, `nonce`) puis oubliés |
| Cookie de session | navigateur | identifiant aléatoire, `HttpOnly`, `SameSite=Strict`, `Secure` en HTTPS |

## Droits

Rôles et permissions dans [roles.toml](../apps/api/src/alliance_api/roles.toml) (ou `ROLES_FILE`),
vérifiés au démarrage ; le code ne teste que des permissions.

| Permission | `manager` | `admin` |
|---|---|---|
| `brief:read_own`, `brief:generate_own`, `feedback:own`, `item:read_own` | ✓ | ✓ |
| `audit:read` (journal, chaîne d'un brief, métriques : identifiants seulement) | | ✓ |
| `brief:read_any` (lire les briefs des autres) | | attribuée à personne |

## Données conservées et durées

| Donnée | Où | Durée | Effacée par `delete-user` |
|---|---|---|---|
| Items (titres, extraits ≤ 500 caractères, participants) | `/data/items/<user>/<jour>/` | 30 jours (`purge`) | oui |
| Corps complet des messages | jamais stocké | — | — |
| Journal de collecte | `/data/items/<user>/audit.jsonl` | 90 jours (`purge`) | oui |
| Journal de l'API | `/data/items/<user>/api-audit.jsonl` | sans limite | oui |
| Briefs générés | `/data/briefs/<user>/` | sans limite | **non** |
| Retours utile / pas utile / faux | `/data/feedback/<user>.jsonl` | sans limite | **non** |
| Sessions | mémoire du processus | 10 h au plus, 1 h sans activité | — |
| Jetons OAuth | `TOKENS_DIR` | jusqu'à révocation | oui |

Durées réglables dans `collect.toml` (section `[retention]`). Les trois lignes « sans limite »
et les deux « non » sont des écarts connus, repris dans [limites.md](limites.md).

## Contrôles en place

| Risque | Mesure | Vérification |
|---|---|---|
| Accès sans session, ou aux données d'un autre | 401 / 403 sur toutes les routes de données (ouvertes : `/health`, connexion, fichiers statiques) | `apps/api/tests/test_auth_errors.py`, `test_permissions.py`, `test_items.py` |
| Requête d'un autre site (CSRF) | `SameSite=Strict` + contrôle `Origin`/`Referer` | `test_permissions.py` |
| Script dans un mail (XSS) | échappement Jinja, aucun filtre `safe` sur une donnée d'item | `test_ui.py::test_script_d_un_mail_echappe_et_non_injecte` |
| Injection de prompt | titres seulement dans le brief, garde déterministe des claims | lot 2, `guard/check.py` |
| Fuite du jeu de test caché | `eval/hidden` hors Git et hors image (`.dockerignore`) | CI : canari absent de l'image |
| Secret commité | gitleaks | pre-commit et CI |
| Source en panne ou lente | réessais avec `Retry-After`, une source n'arrête pas les autres | `packages/collect/tests/test_robustesse.py` |
| Lecture des données d'un autre utilisateur à la collecte | `owner_user_id` contrôlé, noms de dossiers contrôlés | `packages/collect/tests/test_securite.py` |

## À faire avant une mise en production

- HTTPS (proxy TLS) et `ALLOWED_ORIGINS` ; aujourd'hui l'application écoute en HTTP.
- Signature de la revue de sécurité par le référent du lot 1 (L3.4).
- Effacement complet d'un utilisateur et durées de conservation pour les briefs, les retours et
  le journal de l'API.
- Rôles tirés des groupes Entra plutôt que de `roles.toml`.
