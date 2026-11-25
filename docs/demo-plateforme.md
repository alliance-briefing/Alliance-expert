# Démonstration : partie plateforme (L3.10)

La partie « plateforme » de la démonstration finale (installation, sécurité, administration,
données). Le script complet sur trois journées et le plan B général sont dans #80 ; ce document
couvre ce que le lot 3 plateforme présente et défend. Durée : 6 minutes environ.

Journée utilisée : **08/10/2026** de `u_demo_001`, une journée du jeu de développement déjà
annotée (le guide interdit de montrer une journée d'évaluation non annotée).

## Avant le jour J

À faire la veille, sur la machine de démonstration (pas celle du développement) :

- [ ] Cloner depuis le tag : `git clone --branch v1.0.0 https://github.com/alliance-briefing/Alliance-expert.git`
- [ ] Construire l'image (seule étape qui demande le réseau, 1 à 3 minutes) :
      `docker compose -f infra/docker-compose.yml build`
- [ ] Couper le Wi-Fi, puis `docker compose -f infra/docker-compose.yml up -d --wait` : l'application
      doit démarrer et générer le brief du 08/10 sans réseau.
- [ ] Repartir de données neuves pour le jour J : `docker compose -f infra/docker-compose.yml down -v`
      (l'amorçage recrée les 30 journées au démarrage suivant).
- [ ] Ouvrir Git Bash dans le dossier du dépôt (pour `infra/sauvegarde.sh`).
- [ ] Enregistrer la vidéo du plan B en suivant ce script.

## Script

| Temps | Montrer | Dire |
|---|---|---|
| 0:00 | `docker compose -f infra/docker-compose.yml up -d --wait` puis `docker compose -f infra/docker-compose.yml logs api` : la ligne « amorçage : 30 journée(s) ajoutée(s) » | Une seule commande, sans fichier de configuration ni compte externe ; au premier démarrage, la collecte de démonstration du lot 1 remplit le volume. |
| 0:45 | http://localhost:8000, connexion `u_demo_001`, jour 08/10/2026, « Générer le brief » | Le brief retient 7 éléments sur 160. Chaque entrée renvoie à sa source : montrer l'adresse du lien au survol (domaine fictif `.test`, ne pas cliquer). |
| 1:45 | Toujours en `u_demo_001`, ouvrir http://localhost:8000/admin : « Droits insuffisants » (403). Dans le terminal : `curl -i http://localhost:8000/briefs/2026-10-08` → 401 | Le code ne teste jamais un rôle mais une permission (`roles.toml`) ; sans session, aucune donnée ; un manager ne voit que ses propres briefs. |
| 2:30 | Se déconnecter, se connecter avec `u_admin_001`, ouvrir http://localhost:8000/admin?user=u_demo_001&brief=2026-10-08 | La chaîne du brief : collecte (connecteur, source, heure, éléments cités), génération (modèle, durée), consultations (qui, quand). Jamais le texte du brief : l'admin ne lit pas les briefs des autres. |
| 3:30 | Plus bas : « Métriques » et le journal ; puis http://localhost:8000/admin/metrics?days=60 | Taux d'erreur par action et par connecteur, durées de génération. Les durées de collecte ne sont pas mesurées : on le dit plutôt que de les inventer. |
| 4:15 | `docker compose -f infra/docker-compose.yml restart api`, recharger le brief du 08/10 | Le brief est toujours là : les données vivent dans un volume, pas dans le conteneur. |
| 4:45 | `infra/sauvegarde.sh demo.tgz` | Sauvegarde du volume en une commande, sans rien installer ; la restauration est testée à chaque PR par la CI. |
| 5:30 | Rien à montrer | Tout tourne hors ligne : le modèle est dans l'image, les données sont synthétiques ; une source en panne n'arrête pas les autres. |

## Plan B technique

| Si | Alors |
|---|---|
| Pas de réseau dans la salle | Rien à faire : l'image est construite la veille, l'application démarre hors ligne. Seule la police de l'interface change (police système). |
| Docker ne démarre pas | Lancer sans Docker (dépendances installées la veille avec `uv sync --all-packages`), voir ci-dessous. |
| La machine ne démarre pas, ou rien ne marche | Passer la vidéo enregistrée la veille. |

Sans Docker, dans PowerShell, depuis le dossier du dépôt :

```powershell
$env:AUTH_MODE = "demo"
$env:DATA_DIR = "$env:TEMP\briefing\items"; $env:AUDIT_DIR = $env:DATA_DIR
$env:BRIEFS_DIR = "$env:TEMP\briefing\briefs"; $env:FEEDBACK_DIR = "$env:TEMP\briefing\feedback"
uv run python -m alliance_api.seed
uv run uvicorn alliance_api.main:app --port 8000
```

La sauvegarde (`infra/sauvegarde.sh`) n'existe qu'avec Docker : dans ce mode, sauter l'étape 4:45.

## Le tag v1.0.0

La démonstration tourne depuis le tag `v1.0.0` (guide L3.10). À poser par un référent une fois
toutes les PR de la version fusionnées et la CI de `main` verte :

```bash
git switch main && git pull
git tag -a v1.0.0 -m "Version de la démonstration finale"
git push origin v1.0.0
gh release create v1.0.0 --title "v1.0.0" --notes "Démonstration finale : voir docs/exploitation.md"
```

Ensuite, refaire la liste « Avant le jour J » depuis ce tag, sur la machine de démonstration.
