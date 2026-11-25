# Fiche du modèle de briefing (lot 2, M6 à M9)

Version du 25/11/2026 · référent : Théo Szappanyos · décisions : `docs/adr/0005-modele-multi-sources.md`
· plan et règles écrites avant chaque mesure : `packages/analyse/plan-modele.html` (§6, §11).

## À quoi il sert
Classer les éléments du jour d'un manager (mails, réunions, tâches, fichiers) et en tirer son
brief du matin : quoi montrer, dans quel ordre, avec quelle action. Entièrement local, sans API
extérieure, sur processeur. Mode par défaut du brief : **règles** (`ALLIANCE_SCORE=regles`) ;
le modèle s'active avec `ALLIANCE_SCORE=appris`.

## Ce qu'il fait, en trois étages
| étage | modèle | entrée | sortie | fichier |
|---|---|---|---|---|
| comprendre (M6) | MiniLM multilingue gelé + régression logistique, numpy | carte de l'élément : texte normalisé (vecteur de 384 valeurs) et fiche de 39 valeurs lues sur la carte | action (7 classes), urgence (3), alerte (oui/non) | `scoring/ml/comprendre_candidat.npz` |
| prioriser (M8) | Bradley-Terry linéaire, l2 = 0,1 | fiche de priorité : fiche de M6, échéance du jour, 9 probabilités de N | score de priorité, raisons = plus fortes contributions | `scoring/ml/prioriser_modele.json`, `prioriser_echelle.json` |
| sélectionner (M9) | règles réglées sur journées simulées | scores des affaires (une affaire = son meilleur élément) | 3 à 7 entrées, sections, action recommandée par gabarit | `brief.py` |

Jamais noté ni cité : une tentative d'injection (`guard/injection.py`, règle déterministe) et un
envoi en masse (`scoring/ml/noise.py`). Le contenu d'un élément n'est jamais recopié dans le
brief : textes et actions recommandées ne reprennent que les titres.

## Données
- **Entraînement de la compréhension** : 23 390 cartes des 36 personas « train » du simulateur
  (`packages/collect/trainset/`, 12 métiers), écrites par gabarits et, pour les mails, réécrites
  par un LLM local (Gemma 2 9B, consigne mail-v8) ; étiquettes par construction, suivant le guide
  d'annotation (`scoring/annotations/guide-annotation.md`).
- **Entraînement de la priorité** : 300 paires d'éléments du même brief (personas « train »),
  jugées par **Claude** (décision du 21/11 : personne n'avait le temps d'annoter), avec des
  critères écrits avant dans le guide (section 4) ; tour 1 au hasard, tours 2 et 3 sur les paires
  incertaines ; 120 paires « équivalentes ».
- **Développement** (choix des leviers, jamais de mesure finale) : validation dure du simulateur
  (métiers, gabarits et écrivain tenus à l'écart) et jeu réel de 104 cartes (journée de Claire
  écrite à la main, jeu Graph de David), jugées par Claude, échantillon rejugé par Théo (83 %).
- **Mesure, jeux gelés**, jugés par **Théo seul**, sans proposition d'un modèle : T-gen (4
  métiers jamais vus : 150 éléments, 150 paires, 50 paires d'affaires), T-démo (60 éléments et
  40 paires de la démo du lot 1) ; jeu de développement du lot 3 (8 annotations de Davy et
  Guillaume sur 6 journées de démo, guide L3.5 v0.3).

## Résultats (une seule lecture par modèle et par tâche, `scoring/annotations/journal-geles.jsonl`)
| mesure | modèle | référence de comparaison | acceptation |
|---|---|---|---|
| action, F1 macro, T-gen | N **0,769** [0,675 ; 0,845] | référence 0,625 (+0,145, significatif) | — |
| action, F1 macro, T-démo | N **0,270** [0,189 ; 0,364] | référence 0,164 (+0,106, significatif) | — |
| paires d'éléments, T-gen | **0,806** [0,734 ; 0,871] | règles 0,745 (+0,061, non significatif) | ≥ 0,80 et ≥ règles + 0,15 : **non** |
| paires d'éléments, T-démo | **0,703** [0,568 ; 0,838] | règles 0,459 (+0,243, significatif) | rapporté |
| paires d'affaires, T-gen | 0,600 [0,467 ; 0,733] | règles 0,533 (+0,067, non significatif) | pas d'écart négatif : oui |
| brief, NDCG@7, lot 3 | 0,382 | règles 0,275 (+0,107, non significatif) | supérieur aux règles |
| brief, rappel des « doit figurer », lot 3 | 0,554 | règles 0,763 (−0,210, significatif) | **non** |
| brief, « ne doit pas figurer » dans le top 3 | 1 annotation sur 8 | règles 0 | **non** |
| latence par brief, processeur (30 journées de démo) | médiane 2,26 s, 95e centile **7,28 s**, max 10,35 s ; premier brief 2,77 s | règles : 0,01 s | ≤ 3 s : **non** pour le mode appris |

## Limites et biais connus
- **Décalage de domaine** : le modèle apprend sur un simulateur. La compréhension passe de 0,85
  (métiers jamais vus du simulateur) à 0,45 (texte réel) et 0,27 (démo).
- **Jugement de Claude à l'entraînement, de Théo à la mesure** pour la priorité : sur T-gen, le
  modèle perd contre les règles sur les paires mail / tâche (0,756 contre 0,854), où Théo donne
  plus de poids au retard d'une tâche que les critères de Claude.
- **« Déjà traité » invisible** : un élément traité avant le brief ne se voit pas dans l'`Item`
  (contrat A) ; le modèle peut le remonter.
- **Petits jeux de mesure** : 37 à 150 paires, 8 annotations de brief ; les intervalles sont
  larges, plusieurs écarts restent dans le bruit.
- **Relations des correspondants** : configurées pour la démo, sinon déduites (adresse,
  domaine) ; effet mesuré sur la démo : 94 % des 7 premières affaires conservées.
- **Alertes** : une alerte de T-gen sur deux manquée par N (la référence trouvait les deux) ;
  l'alerte de T-démo manquée par tous les modèles. Les alertes prévues sont tenues en tête du brief.
- **Réglage l2 = 0,1 au bord de la grille** de Bradley-Terry, noté, grille non élargie.
- **Latence du mode appris** : au-delà de 3 s sur les journées chargées de la démo (95e
  centile 7,3 s), parce que chaque élément au brief est encodé par MiniLM ; le mode par défaut
  (règles) répond en 0,01 s.

## Reproduire
```bash
uv run python -m packages.analyse.scoring.ml.comprendre --figer-candidat      # N, contrôle 0,445
uv run python -m packages.analyse.scoring.ml.prioriser --valider              # priorité, hors pli
uv run python -m packages.analyse.scoring.ml.prioriser --figer                # BT contre boosting
uv run python -m packages.analyse.scoring.ml.prioriser --echelle              # échelle 0-100
uv run python -m packages.analyse.scoring.ml.selection --valider              # réglage de T et K
uv run python -m packages.analyse.scoring.ml.selection --latence              # latence (M10)
```
Les mesures sur les jeux gelés (`--mesurer`, `--mesurer-candidat`, `--mesurer-affaires`,
`--mesurer-lot3`) refusent une seconde lecture du même modèle : leurs chiffres se relisent dans le
journal, avec le commit et l'empreinte du modèle (étiquettes `m6-mesure-N` et `m9-mesure` quand le
commit a été réécrit par une fusion).
