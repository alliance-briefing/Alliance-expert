# Démonstration commentée : le brief du mercredi 7 octobre (démo du lot 1)

Camille Laurent, responsable d'une agence d'expertise automobile, 192 éléments ce jour-là. Cette
journée n'appartient pas au jeu de développement du lot 3 : elle n'a servi à aucune mesure.

```bash
uv run python -m packages.analyse.scoring.ml.selection --demo 2026-10-07
```

## Le brief par règles (mode par défaut)
12 entrées : 7 actions, toutes les réunions du jour (5), et les tâches en retard s'il y en a.
Les scores se départagent mal : six actions à 70 (« marqué comme important », « échéance dans
moins de 24 h »), les réunions à 50 par construction, triées par heure. Rien ne dit quoi faire.

## Le brief appris (`ALLIANCE_SCORE=appris`)
7 entrées, le maximum que demande le guide du lot 3 :

| rang | entrée | ce que fait le modèle |
|---|---|---|
| actions 1 | Tâche « Traiter les réclamations clients en attente » (70) | N lit « action demandée » ; l'action recommandée reprend l'intitulé de la tâche, déjà une action |
| actions 2-3 | Deux relances « Mission d'expertise – … – RELANCE » (68) | N lit « réponse attendue » et un ton urgent ; le reste de l'affaire (mail, fichier) est cité en contexte |
| actions 4 | « Facture garage – dossier 2026-3204 » (68) | N lit « décision ou validation attendue » → « Décider ou valider : … » |
| réunions 1-2 | Deux réunions de 11 h (68) | seules les réunions du jour qui comptent entrent ; la formation RGPD toute la journée et la revue qualité de 14 h n'entrent pas |
| retards 1 | « Affecter un expert au dossier 2026-3025 » (80) | tâche en retard et action demandée ; la réunion du dossier à 14 h 15 est citée en contexte |

Ce qu'il faut regarder :
- **Les raisons sont celles du modèle** : les plus fortes contributions de la fiche de priorité
  (Bradley-Terry), écrites en clair ; une probabilité de N n'est une raison que si elle dépasse
  0,5.
- **Les scores sont des rangs centiles** : 70 veut dire « passe avant 70 % des éléments
  d'entraînement » ; 50, l'élément médian, est le seuil d'entrée.
- **Aucun contenu n'est recopié** : textes et actions ne reprennent que des titres ; une
  tentative d'injection ou une lettre d'information n'est jamais notée.

## Ce que la mesure en dit
Sur les 8 annotations du lot 3, le brief appris ordonne mieux (NDCG@7 0,382 contre 0,275) mais
retrouve moins d'éléments attendus (rappel 0,554 contre 0,763), parce qu'il s'arrête à 7
entrées quand le brief par règles en montre 11 à 15. Les règles restent donc le défaut, et le
mode appris dépasse 3 s par brief sur les journées chargées (95e centile 7,3 s). Détails :
`fiche-modele.md` et `rapport-ablations.md`.
