# Authentification, sessions et droits (L3.4)

Qui peut entrer dans l'application, combien de temps, et ce qu'il peut y faire.
Code : `apps/api/src/alliance_api/` — `auth.py`, `oidc.py`, `sessions.py`, `resume.py`,
`permissions.py`, `roles.toml`.

> La connexion des **utilisateurs à l'application** (ce document) est distincte de
> l'accès de la **collecte** à leurs mails et agenda (`oauth-microsoft.md`, lot 1).

## Modes (`AUTH_MODE`)

| Mode | Usage | Connexion |
|---|---|---|
| `demo` | local, CI, démo hors ligne, données synthétiques | identifiant au choix : **aucune sécurité** |
| `oidc` | démo avec de vrais comptes, version finale | fournisseur OpenID Connect (Microsoft pour la démo) |
| vide | — | API fermée (401) |

Les routes ne dépendent que de « qui est connecté » et de « a-t-il la permission » :
changer de mode ou de fournisseur ne touche pas le reste du code.

## Connexion Microsoft (OpenID Connect)

1. Dans Entra, sur l'application de démo, ajouter l'URI de redirection
   `http://localhost:8000/auth/callback` (plateforme « Applications mobiles et de bureau » :
   client public, PKCE, pas de secret ; ou plateforme « Web » avec `OIDC_CLIENT_SECRET`).
2. Copier `infra/users.example.toml` **hors du dépôt**, y mettre les comptes autorisés
   (`"<tid>:<oid>" = "u_…"`, identifiants visibles dans le portail Entra).
3. Lancer :
   ```sh
   AUTH_MODE=oidc OIDC_ISSUER=https://login.microsoftonline.com/common/v2.0 \
   OIDC_CLIENT_ID=<id d'application> USERS_FILE=<chemin>/users.toml \
   uv run uvicorn alliance_api.main:app
   ```
   puis ouvrir <http://localhost:8000/auth/login>.

Une variable manquante ou un fichier de comptes invalide empêche le démarrage.

**Changer de fournisseur** (Google, Okta, Keycloak, un autre locataire Entra…) : changer
`OIDC_ISSUER`, `OIDC_CLIENT_ID`, éventuellement `OIDC_CLIENT_SECRET`, et
`OIDC_IDENTITY_CLAIMS` (souvent `sub` hors Microsoft). Pour n'accepter qu'un locataire
Entra : `OIDC_ISSUER=…/<id-du-locataire>/v2.0` ou `OIDC_TENANTS=<id>`.

## Sessions et reprise du travail

- Session côté serveur : le cookie `session` ne contient qu'un identifiant aléatoire
  (HttpOnly, SameSite=Strict, Secure en HTTPS).
- **10 h** maximum après la connexion, **1 h** sans activité (`SESSION_MAX_HOURS`,
  `SESSION_IDLE_MINUTES`). Déconnexion : la session est effacée côté serveur.
- Expiration pendant un travail : l'action (générer le brief, donner un retour) est mise de
  côté 15 min ; après reconnexion **de la même personne**, elle est rejouée et la personne
  revient sur sa page. Les clients d'API reçoivent 401 `session_expired`.

## Droits

`roles.toml` (ou `ROLES_FILE`) associe chaque rôle à des permissions ; le code ne teste que
des permissions. Par défaut, personne ne lit les briefs des autres, pas même l'admin :
la permission `brief:read_any` existe et peut être attribuée si l'entreprise le demande.

## Contrôles de sécurité

| Risque | Protection | Test |
|---|---|---|
| Code d'autorisation intercepté | PKCE S256 | `test_connexion_complete` |
| Connexion imposée par un tiers | `state` lié au cookie du navigateur, usage unique, 10 min | `test_retour_dans_un_autre_navigateur_refuse`, `test_state_inconnu_ou_rejoue` |
| Jeton rejoué ou forgé | signature (clés publiées), `aud`, `iss` du locataire, `exp`, `nonce` | `test_oidc.py` |
| Compte Microsoft quelconque | seuls les comptes de `USERS_FILE` entrent | `test_compte_absent_du_fichier` |
| Requête d'un autre site (CSRF) | SameSite=Strict + contrôle Origin/Referer | `test_permissions.py` |
| Redirection vers un autre site | chemins internes uniquement | `test_redirection_externe_ignoree` |
| Fuite de jetons | aucun jeton gardé ni journalisé (seul le code d'erreur OAuth) | — |

## Limites (version de démonstration)

- Sessions et actions en attente en mémoire : un redémarrage déconnecte tout le monde
  (aucune donnée perdue, il suffit de se reconnecter).
- La déconnexion ferme la session de l'application, pas celle de Microsoft.
- Le rôle vient de `roles.toml`, pas des groupes Entra.

À relire et signer par le référent sécurité du lot 1 avant la version finale.
