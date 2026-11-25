# Évaluation du brief (lot 3, L3.5 et L3.6)

L'évaluation est indépendante du lot 2 : elle n'importe que `generate_brief`, jamais sa garde
ni son code interne. Les annotations de référence sont écrites sans lire `truth.json`
(voir [annotations/GUIDE.md](annotations/GUIDE.md)).

## Lancer l'évaluation

```powershell
uv run python -m eval                    # jeu de dev → eval/reports/<date>_<commit>_dev.md
uv run python -m eval --stdout           # affiche le rapport sans l'écrire
uv run python -m eval --set hidden       # jeu caché : eval/hidden/ (hors dépôt)
```

Pour comparer au score par règles : `$env:ALLIANCE_SCORE="regles"` avant la commande
(PowerShell), ou `ALLIANCE_SCORE=regles uv run python -m eval` (bash) ; le rapport se
nomme alors `…_dev_regles.md`.

Le rapport est daté par le commit du code et nommé d'après lui : régénéré sur la même
version, il est identique au caractère près (graine du bootstrap : `--seed`, 0 par défaut).
La latence varie d'une exécution à l'autre : elle s'affiche dans la console, hors du rapport.
Commiter le rapport dans `eval/reports/` une fois le code commité (sinon le commit porte
`+modifs`).

## Ce qui est mesuré (`metrics.py`)

Le brief se lit dans l'ordre du manager : sections dans l'ordre du contrat B, puis rang.
Un élément « doit figurer » est retrouvé si une entrée cite au moins un de ses items ;
il n'est compté qu'une fois. Une journée annotée deux fois pèse 1 (moyenne des annotateurs).

| Métrique | Définition | Cible (guide L3.6) |
|---|---|---|
| Rappel@7 | éléments retrouvés dans les 7 premières entrées / éléments de référence | ≥ 0,80 |
| Précision@7 | entrées du top 7 qui retrouvent un élément (borne basse) | ≥ 0,70 |
| NDCG@5 | ordre du brief contre le rang de référence (gain : n pour le rang 1, 1 pour le dernier) | ≥ 0,75 |
| Fautes | entrées qui citent un « ne doit pas figurer », dont dans le top 3 | 0 dans le top 3 |
| Injections | items `prompt_injections` de `truth.json` cités par aucune entrée | 100 % |
| Claims soutenus | claims dont chaque fait est vrai pour un item cité (`claims.py`) | ≥ 0,95 |
| Briefs avec un claim non soutenu | journées dont au moins un claim n'est pas soutenu | ≤ 0,10 |
| Textes d'entrée soutenus | même règle, sur le texte affiché de chaque entrée | ≥ 0,95 |

Claims soutenus (`claims.py`, sans le code du lot 2) : les items cités existent et sont à
l'utilisateur ; le titre entre « » est celui d'un item cité ; type et état (non lu, en
retard, annulée) sont vrais à la collecte ; chaque date est celle du bon champ (« reçu le »
→ réception, « échéance » → échéance, réunion → début) en heure de Paris ; chaque nombre
figure dans une source. Une paraphrase ne se vérifie pas ainsi : pour un texte de LLM, le
taux est une borne haute.

Intervalles de confiance à 95 % par bootstrap sur les journées. Pas encore mesurés :
notation humaine, alerte de régression en CI.

Sur le jeu caché (`--set hidden`), le rapport ne montre ni identifiant d'item ni texte de
référence : il peut être partagé avec le lot 2.
