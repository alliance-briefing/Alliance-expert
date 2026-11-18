# Journée réelle de démonstration (lot 1)

`u_claire/2026-11-19/items.jsonl` : 34 Items (contrat A 1.1) collectés **pour de vrai** chez
Microsoft par les connecteurs du lot 1, sur un compte de démonstration **entièrement fictif**
(persona Claire Fontaine, consultante ; Victor Mabe, client ; Titouan Synayoko, associé).
Aucune donnée réelle : tous les contenus ont été rédigés pour le scénario.

- Collecte du 19/11/2026 à 8h30 (Paris) : 22 mails, 3 réunions, 4 tâches To Do, 5 fichiers OneDrive.
- Les 6 mails automatiques de Microsoft (alertes de sécurité, bienvenue OneDrive) ont été retirés.
- Scénario : copil du 18/11, slides à envoyer, facture bloquée, demande urgente, question du
  client sans réponse, invitations reliées aux réunions (`correlation_id`) ; et du bruit
  (rappel de saisie des temps, déjeuner d'équipe, formation facultative).

## Utiliser cette journée

```sh
AUTH_MODE=demo DATA_DIR=packages/collect/demo docker compose -f infra/docker-compose.yml up --build
# puis se connecter avec u_claire et choisir le 19/11/2026
```

Avec `DATA_DIR` défini, l'API ne lit plus les journées de `contracts/fixtures/demo`.
Les liens (`url`) ouvrent les éléments dans le compte de démo : ils ne marchent qu'une fois
connecté à ce compte.
