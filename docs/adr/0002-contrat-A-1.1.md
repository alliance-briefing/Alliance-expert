# ADR-0002 : contrat A 1.1 — `correlation_id` et `attributes`
Date : 2026-10-16 · Statut : proposé
Décideurs : référents lot 1, 2, 3

## Contexte
Pour le jalon v0.2.0 (21/10), le lot 1 branche les vrais mails et le vrai calendrier. Deux informations exigées par le guide n'ont pas de place dans le contrat A 1.0 :
- **L1.6** : une invitation reçue par mail et l'événement du calendrier doivent partager un **identifiant de corrélation** (sinon le brief montre deux fois la même réunion) ;
- **L1.4** : lieu, lien de visio, « toute la journée », annulé, réponse (accepté/refusé/provisoire). Aujourd'hui, ils finiraient dans `snippet`, en texte libre, inutilisables de façon fiable par le lot 2.

## Options examinées
1. **Ne rien changer** : le lot 2 devine à partir du texte. Coût nul, mais « annulé » ou « toute la journée » ne sont pas fiables, et rien ne relie l'invitation à la réunion.
2. **Champs à plat** (`location`, `is_all_day`, …) : lisible, mais le contrat grossit à chaque source et les champs n'ont de sens que pour le calendrier.
3. **Deux champs optionnels : `correlation_id` + un bloc `attributes` typé par source** (recommandé). Une seule évolution mineure, extensible.

Test sur les données de David (`alliance-graph-synth`) : avec la règle « titre sans “Annulé :” + heure de début », **les 70 invitations sur 70** retrouvent leur événement.

## Décision
Contrat A **1.1** (ajout de champs optionnels = version mineure, guide §2.5) :

```json
"correlation_id": "meet:point-flash-operations:2026-10-21T07:00Z",
"attributes": {
  "calendar": {
    "location": "Salle Atlas",
    "online_url": "https://teams.microsoft.com/...",
    "is_all_day": false,
    "is_cancelled": false,
    "response": "accepted",
    "series_id": "AAMk..."
  }
}
```

- `correlation_id` : texte ou `null`, rempli par la normalisation (L1.6), identique sur l'invitation et l'événement.
- `attributes` : `null` par défaut ; s'il est présent, il contient un seul bloc, celui de la source de l'Item (`calendar` pour `source = "calendar"`). Les clés sont **fixées et typées** (champ inconnu refusé, comme le reste du contrat). Un bloc `file` (chemin, taille, type) pourra être ajouté pour L1.5 en 1.2.
- `schema_version` accepte `"1.0"` et `"1.1"` : les fixtures existantes restent valides sans régénération.

## Conséquences
- **Lot 1** : la normalisation remplit `correlation_id`, le connecteur calendrier remplit `attributes.calendar`. PR `contract-change` avec JSON Schema, fixtures valides/invalides, tests et CHANGELOG.
- **Lot 2** : peut regrouper par `correlation_id` et écarter les réunions annulées sans lire le texte. Rien à changer s'il ignore ces champs.
- **Lot 3** : peut afficher le lien de visio ; l'évaluation peut vérifier qu'une réunion n'apparaît qu'une fois.
- **Retour arrière** : les deux champs sont optionnels ; les retirer revient au contrat 1.0 sans casser les données.
