# Rapport d'ablations (plan, §6) — lot 2, M6 à M9

Version du 25/11/2026. Chaque chiffre vient d'un rapport versionné ou du journal des jeux gelés,
et se reproduit par la commande indiquée (depuis la racine du dépôt, `uv run python -m …`).
Les décisions suivent les règles G1 à G6 du plan (§6.1) : choix sur la validation, intervalle
apparié à 1 000 tirages (G2), à égalité le plus simple (G3), résultat négatif publié (G6).
Les mesures « dév. » servent à choisir ; seules les lignes « gelé » mesurent, une fois.

## 1. Règles → MiniLM gelé → encodeur réajusté (compréhension, F1 macro de l'action)
| variante | métiers jamais vus (dév.) | jeu réel, 104 cartes (dév.) | T-gen (gelé) | démo (gelé) |
|---|---|---|---|---|
| référence : MiniLM gelé + régression | 0,728 | 0,229 | 0,625 | 0,164 |
| + fiche lue sur la carte (H) | 0,804 | 0,382 | — | — |
| + texte normalisé (**N**, retenu) | 0,851 | 0,445 | **0,769** | **0,270** |
| MiniLM réajusté de bout en bout (E0 / E1) | 0,945 / 0,946 | 0,299 / 0,341 | — | — |

Commandes : `packages.analyse.scoring.ml.comprendre --valider-dur`, `--levier-fiche`,
`--levier-texte` ; `packages.analyse.scoring.ml.encodeur --levier` (carte graphique).
Rapports : `comprendre_validation_dure.json`, `comprendre_levier_fiche.json`,
`comprendre_levier_texte.json`, `comprendre_levier_encodeur.json`. Lecture : chaque gain qui ne
dépend pas du style (fiche, normalisation) transfère ; le réajustement apprend la formulation du
simulateur et perd sur le texte réel (−0,146 [−0,225 ; −0,042] pour E0).

## 2. Gabarits seuls → gabarits + LLM (écrivain croisé, F1 macro de l'action, dév.)
| appris sur → mesuré sur | référence | H | N |
|---|---|---|---|
| gabarits → gabarits | 0,751 | — | 0,880 |
| gabarits → LLM | 0,501 | 0,571 | 0,634 |
| LLM → gabarits | 0,524 | 0,620 | 0,679 |
| LLM → LLM | 0,674 | — | 0,822 |
| gabarits tenus à l'écart | 0,578 | 0,683 | 0,716 |

Commandes : `comprendre --valider-dur`, `--levier-texte`. Lecture : changer d'écrivain coûte 0,15 à
0,25 ; les deux écrivains ensemble sont la base de tout l'entraînement. Un second écrivain LLM
(levier 4 du plan d'amélioration) n'a pas été fait : la mesure finale de N lui a été préférée.

## 3. Augmentation du texte (levier 2, dév.)
| variante | métiers | gabarits tenus à l'écart | jeu réel |
|---|---|---|---|
| H | 0,804 | 0,683 | 0,382 |
| **N, texte normalisé** | **0,851** | **0,716** | **0,445** |
| A, copies perturbées | 0,810 | 0,681 | 0,445 |
| N + A | 0,857 | 0,726 | 0,440 |

Commande : `comprendre --levier-texte`. A et N + A ne battent pas N (arrêt à la graine 0).

## 4. Priorité : règles → Bradley-Terry → boosting (accord sur les paires départagées)
| | règles (`score_case`) | **Bradley-Terry** | boosting de souches |
|---|---|---|---|
| 300 paires de Claude, hors pli par métier (dév.) | 0,722 | **0,928** | 0,756 |
| T-gen, 139 paires de Théo (gelé) | 0,745 | **0,806** | — |
| T-démo, 37 paires de Théo (gelé) | 0,459 | **0,703** | — |
| T-gen, 45 paires d'affaires (gelé) | 0,533 | 0,600 | — |

Commandes : `packages.analyse.scoring.ml.prioriser --valider`, `--figer`. Rapports :
`prioriser_validation.json`, `prioriser_comparaison.json`. Acceptation de M8 (T-gen ≥ 0,80 et
≥ règles + 0,15) **non atteinte** : +0,061 non significatif ; perte sur mail / tâche.

## 5. Apprentissage actif contre hasard, à budget égal — non fait
Les tours 2 et 3 de la priorité ont été tirés sur les paires incertaines, mais aucun témoin au
hasard à budget égal n'a été annoté (personne n'avait le temps ; Claude a annoté une seule
série). Constat sans comparaison : la sélection incertaine trouve surtout de vraies égalités (48
et 60 « équivalentes » sur 100, contre 12 au hasard), et l'écart aux règles a grandi de +0,114
(tour 1) à +0,206 (trois tours) sur les paires de développement.

## 6. Liaison apprise des affaires, historique retrouvé (RAG) — non faits
Reportés en M8 (G3 : une brique de moins à défendre) ; la liaison reste celle des règles L2.1.
Aucune mesure : on ne peut rien conclure sur leur apport.

## 7. Transfert : simulateur → T-gen → démo → jeu réel
| | compréhension N (F1 action) | priorité (accord) |
|---|---|---|
| simulateur, métiers jamais vus (dév.) | 0,851 | 0,928 (jugement de Claude) |
| T-gen, métiers réservés (gelé) | 0,769 | 0,806 (jugement de Théo) |
| démo du lot 1 (gelé) | 0,270 | 0,703 |
| jeu réel, Claire et David (dév.) | 0,445 | — |

Le transfert vers le vrai domaine reste la faiblesse principale de la compréhension ; la
priorité transfère mieux, parce qu'elle combine N avec des signaux lisibles (retard, réunion du
jour, expéditeur). Le jeu de David n'a pas servi de test séparé (une partie forme le jeu réel).

## 8. Brief : règles contre sélection apprise (M9)
| | brief par règles | brief appris (T = 50, K = 7) |
|---|---|---|
| 60 journées simulées, NDCG@7 (dév.) | — | 0,347 |
| lot 3, NDCG@7 (gelé) | 0,275 | **0,382** (+0,107, non significatif) |
| lot 3, rappel des « doit figurer » (gelé) | **0,763** | 0,554 (−0,210, significatif) |
| lot 3, « ne doit pas figurer » dans le top 3 | 0 | 1 annotation sur 8 |
| entrées par brief | 11 à 15 | 7 |

Commandes : `packages.analyse.scoring.ml.selection --valider`, `--mesurer-lot3`. Acceptation de M9
**non atteinte** : les règles restent le défaut. Le brief appris ordonne mieux mais, limité à 7
entrées comme le demande le guide du lot 3, retrouve moins que le brief par règles plus long.

## 9. Production
Latence par brief sur les 30 journées de démo (`packages.analyse.scoring.ml.selection --latence`
→ `brief_latence.json`, processeur Intel du portable de Théo) :

| mode | premier brief | médiane | 95e centile | max | ≤ 3 s |
|---|---|---|---|---|---|
| règles (défaut) | 0,01 s | 0,01 s | 0,02 s | 0,02 s | oui |
| appris | 2,77 s | 2,26 s | 7,28 s | 10,35 s | **non** |

Le mode appris dépasse le seuil sur les journées chargées : chaque élément au brief est encodé
par MiniLM. Acceptation de M10 tenue pour le mode par défaut, pas pour le mode appris.
