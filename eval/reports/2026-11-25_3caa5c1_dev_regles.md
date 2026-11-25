# Évaluation du brief — jeu « dev » — 2026-11-25

- Code : `3caa5c1` (commit du 2026-11-25)
- Générateur : rules ; `ALLIANCE_SCORE` = `regles`
- Jeu : 6 journée(s), 8 annotation(s) ; graine du bootstrap : 0
- Commande : `uv run python -m eval --set dev --seed 0`

## Métriques

Moyenne par journée (une journée annotée deux fois pèse 1). Intervalle de confiance à 95 % par bootstrap sur les journées. Avec moins de 10 journées, il est large : ne pas conclure sur un écart qui reste dans l'intervalle.

| Métrique | Valeur | IC 95 % | Cible | Verdict |
|---|---|---|---|---|
| Rappel@7 | 0,25 | [0,10 ; 0,40] | ≥ 0,80 | ❌ non atteint |
| Précision@7 | 0,21 | [0,10 ; 0,33] | ≥ 0,70 | ❌ non atteint |
| NDCG@5 | 0,28 | [0,06 ; 0,49] | ≥ 0,75 | ❌ non atteint |
| « Ne doit pas figurer » dans le top 3 | 0 journée(s) sur 6 | — | 0 | ✅ atteint |
| Injections de prompt neutralisées (non citées) | 8/8 | — | 100 % | ✅ atteint |
| Claims soutenus par leur source | non mesuré (étape suivante) | — | ≥ 0,95 | — |
| Notation humaine (clarté, concision, action) | non mesuré | — | ≥ 4,0 | — |

Lecture : ordre du brief = sections dans l'ordre du contrat B, puis rang. Un élément de référence est retrouvé si une entrée cite au moins un de ses items. Précision@7 = part des 7 premières entrées qui retrouvent un élément « doit figurer » : c'est une borne basse, car le guide laisse des éléments ni inclus ni exclus.

## Par journée

| Journée | Annotateur | Entrées | Rappel | Précision | NDCG | Fautes (top 3) |
|---|---|---|---|---|---|---|
| 2026-10-05 | davy-junior | 11 | 0,29 | 0,29 | 0,63 | 0 (0) |
| 2026-10-08 | guillaume-carrier | 13 | 0,00 | 0,00 | 0,00 | 0 (0) |
| 2026-10-09 | davy-junior | 12 | 0,00 | 0,00 | 0,00 | 1 (0) |
| 2026-10-09 | guillaume-carrier | 12 | 0,00 | 0,00 | 0,00 | 0 (0) |
| 2026-10-12 | guillaume-carrier | 14 | 0,40 | 0,29 | 0,08 | 0 (0) |
| 2026-10-13 | davy-junior | 15 | 0,29 | 0,29 | 0,38 | 0 (0) |
| 2026-10-15 | davy-junior | 11 | 0,60 | 0,43 | 0,72 | 0 (0) |
| 2026-10-15 | guillaume-carrier | 11 | 0,43 | 0,43 | 0,52 | 0 (0) |

## Accord entre annotateurs (kappa de Cohen, attendu ≥ 0,60)

- 2026-10-09 : 0,49
- 2026-10-15 : 0,54

## Les 10 pires cas

- **2026-10-08** (guillaume-carrier) — NDCG 0,00, rappel 0,00
  - rang 1 (actions) lu en position 8, hors du top 7 — Traiter au plus vite les dossiers en attente pour ne pas prendre de retard
  - rang 2 (actions) lu en position 12, hors du top 7 — URGENT, tâche en retard
  - rang 3 (meetings) lu en position 9, hors du top 7 — Faire le point sur les données de la semaine passée, ventes, budgets, réclamations...
  - rang 4 (meetings) lu en position 11, hors du top 7 — Point avec un collaborateur
- **2026-10-09** (davy-junior) — NDCG 0,00, rappel 0,00
  - rang 1 (actions) absent du brief — il est important de le faire car les clients attendent
  - rang 2 (meetings) lu en position 8, hors du top 7 — reunion importante
  - rang 3 (overdue) lu en position 12, hors du top 7 — il faut faire le reporting mensuel
  - rang 4 (actions) absent du brief — planning à compléter
  - 1 « ne doit pas figurer » cité(s), dont 0 dans le top 3 : `mail:dm1-msg-000035`
- **2026-10-09** (guillaume-carrier) — NDCG 0,00, rappel 0,00
  - rang 1 (meetings) lu en position 8, hors du top 7 — À préparer avant d'y aller.
  - rang 2 (actions) lu en position 12, hors du top 7 — Pour ne pas avoir plus de retard.
  - rang 3 (meetings) lu en position 9, hors du top 7 — Pour ne pas prendre de retard sur ces dossiers.
  - rang 4 (actions) absent du brief — Pour organiser la journée.
- **2026-10-12** (guillaume-carrier) — NDCG 0,08, rappel 0,40
  - rang 1 (meetings) lu en position 8, hors du top 7 — Détail de tous les dossiers pour être à jour.
  - rang 2 (actions) lu en position 11, hors du top 7 — URGENT, tâches en retard.
  - rang 3 (actions) lu en position 10, hors du top 7 — Éviter un retard.
- **2026-10-13** (davy-junior) — NDCG 0,38, rappel 0,29
  - rang 1 (actions) absent du brief — Une collègue demande de rappeler le client du dossier 2026-7121, qui menace de saisir un avocat faute de réponse.
  - rang 4 (overdue) lu en position 12, hors du top 7 — La validation du devis de 8 300 € pour le dossier 2026-3025 est en retard depuis le 9 octobre ; le rapport est disponible.
  - rang 5 (overdue) lu en position 13, hors du top 7 — Le rapport du dossier 2026-4823 reste à relire et signer depuis son échéance du 9 octobre à 17 h.
  - rang 6 (overdue) lu en position 14, hors du top 7 — Le rapport du dossier 2026-4900 reste à relire et signer depuis son échéance du 9 octobre à 17 h.
  - rang 7 (overdue) lu en position 15, hors du top 7 — La préparation du reporting mensuel, couvrant délais, volumes, contestations et dossiers sensibles, est en retard depuis le 8 octobre.
- **2026-10-15** (guillaume-carrier) — NDCG 0,52, rappel 0,43
  - rang 1 (actions) absent du brief — Le client du dossier 2026-7121 menace de saisir un avocat ; rappel demandé par Anaïs Lefèvre-Dubœuf, sans réponse depuis le 08/10.
  - rang 3 (actions) absent du brief — Devis 2026-9859 de 6 800 € au-dessus du seuil, à valider avant le 14/10 12h (dépassé) ; Mutuelle Horizon Sud attend.
  - rang 4 (actions) absent du brief — Kevin Barbier (dossier 2026-5006) sans nouvelles depuis plus d'une semaine, demande à être rappelé.
  - rang 7 (actions) absent du brief — Reporting d'octobre à transmettre à Bernard Mercier avant le 16/10 midi pour le comité de direction.
- **2026-10-05** (davy-junior) — NDCG 0,63, rappel 0,29
  - rang 2 (meetings) lu en position 9, hors du top 7 — L’expertise sur place du dossier 2026-6598 et la visite au garage partenaire se chevauchent de 10 h 45 à 11 h 15 ; il faut arbitrer.
  - rang 4 (actions) absent du brief — Le client attendait pour sa revue du 2 octobre la synthèse des sinistres du trimestre et des recommandations de prévention.
  - rang 5 (actions) absent du brief — L’assureur attend une date d’expertise pour le dossier 2026-4900, où le choc arrière a déformé le plancher.
  - rang 6 (actions) absent du brief — L’assureur attend une confirmation et une date d’expertise pour le dossier 2026-5455 depuis le 1er octobre.
  - rang 7 (overdue) lu en position 11, hors du top 7 — L’affectation d’un expert au dossier 2026-3156 est en retard depuis le 1er octobre à midi.
- **2026-10-15** (davy-junior) — NDCG 0,72, rappel 0,60
  - rang 2 (meetings) lu en position 8, hors du top 7 — meeting important
  - rang 5 (overdue) lu en position 11, hors du top 7 — entretiens annuels
