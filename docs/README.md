# Documentation technique

Dans l'ordre du guide d'équipe (L3.9). Pour démarrer vite : [README du dépôt](../README.md).

| Section | Document |
|---|---|
| Architecture : diagramme, flux de données, contrats, responsabilités des lots | [architecture.md](architecture.md) |
| Technologies retenues et choix justifiés | [architecture.md, « Technologies retenues »](architecture.md#technologies-retenues) et les [ADR](adr/) |
| Sécurité : menaces, jetons, droits, données conservées et durées | [securite.md](securite.md), [threat-model.md](threat-model.md), [auth.md](auth.md) |
| Évaluation : protocole, jeux, métriques, résultats, limites | à venir (#80) ; jeux et guide d'annotation : [eval/annotations/GUIDE.md](../eval/annotations/GUIDE.md) |
| Guide d'exploitation : installation, configuration, ajout d'une source, changement de modèle | [exploitation.md](exploitation.md) |
| Utilisation | à venir (#80) |
| Limites connues et pistes pour Alliance Experts | [limites.md](limites.md) |

Par lot : [collecte](../packages/collect/README.md) (dont [OAuth Microsoft](oauth-microsoft.md)),
[analyse](../packages/analyse/README.md) (dont la [fiche du modèle](../packages/analyse/docs/fiche-modele.md)),
[contrats](../contracts/README.md).

## Décisions (ADR)

| ADR | Décision | Statut |
|---|---|---|
| [0001](adr/0001-interface-jinja.md) | Interface rendue côté serveur (FastAPI + Jinja) | proposé |
| [0002](adr/0002-contrat-A-1.1.md) | Contrat A 1.1 : `correlation_id` et `attributes` | proposé |
| [0003](adr/0003-score-appris.md) | Score de priorité appris, caractéristiques lisibles | proposé |
| [0004](adr/0004-google-reporte.md) | Fournisseur Google reporté après la v0.9.0 | accepté |
| [0005](adr/0005-modele-multi-sources.md) | Modèle de briefing multi-sources | proposé |
| [0006](adr/0006-stockage-fichiers.md) | Données en fichiers dans un volume, sans Postgres | proposé |
