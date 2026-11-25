# ADR-0005 : modèle de briefing multi-sources (compréhension, priorité, sélection)
Date : 2026-11-25 · Statut : proposé
Décideurs : référent global (Théo Szappanyos), lot 2 ; remplace la partie « mails » d'ADR-0003

## Contexte (3 lignes)
Le brief doit classer les mails, réunions, tâches et fichiers d'un manager avec un modèle
original, local, sans API extérieure (décisions du 01/11, plan `packages/analyse/plan-modele.html`).
Aucune donnée réelle d'entraînement : un simulateur (`packages/collect/trainset/`) et un écrivain
LLM local fournissent l'entraînement ; des jeux gelés jugés par un humain servent à la seule mesure.

## Options examinées
1. **Compréhension (M6)** : MiniLM multilingue gelé + régression logistique (référence), plus
   une fiche lue sur la carte et un texte normalisé (**N**), contre un MiniLM réajusté de bout en
   bout sur carte graphique (**E**). Hors pli par métier : N 0,851, E 0,945 ; mais sur le jeu
   réel N 0,445, E 0,299 et 0,341 : E apprend la formulation du simulateur. **N retenu**
   (levier 3, `comprendre_levier_encodeur.json`).
2. **Priorité (M8)** : Bradley-Terry linéaire sur une fiche de priorité (signaux de la carte et
   probabilités de N), contre un boosting de souches par paires. Hors pli : 0,928 contre 0,756.
   **Bradley-Terry retenu** (`prioriser_comparaison.json`) ; ses poids se lisent comme raisons.
3. **Rédaction** : LLM génératif local écarté en production (latence sur processeur, preuve des
   affirmations) ; textes et actions recommandées par gabarits à partir des titres.
4. **Mode par défaut du brief (M9)** : brief appris (sélection de 3 à 7 entrées) contre brief
   par règles. Sur le jeu du lot 3 : NDCG@7 0,382 contre 0,275 (non significatif), rappel 0,554
   contre 0,763 (significatif). **Règles par défaut** (règle écrite avant la mesure).

## Décision
- Compréhension : N (`comprendre_candidat.npz`) ; priorité : Bradley-Terry
  (`prioriser_modele.json`, échelle `prioriser_echelle.json`) ; affaire = son meilleur élément.
- Brief appris disponible avec `ALLIANCE_SCORE=appris` (sélection T = 50, K = 7, alertes en
  tête, action recommandée, injections et envois en masse jamais cités) ; **`regles` par défaut**.
- Toute mesure sur un jeu gelé passe par une règle écrite avant (plan, §11) et une seule lecture
  inscrite au journal (`packages/analyse/scoring/annotations/journal-geles.jsonl`).

## Conséquences
- Coût : MiniLM en ONNX (fastembed) préchargé dans l'image ; le reste est numpy et quelques Ko de
  poids. Latence sur les 30 journées de démo (`brief_latence.json`) : règles 0,01 s ; appris,
  médiane 2,3 s mais 95e centile 7,3 s, au-delà du seuil de 3 s sur les journées chargées.
- Limites publiées : acceptation de M6 atteinte contre la référence seulement, de M8 et M9 non
  atteinte ; jugement de Claude à l'entraînement de la priorité, jugement de Théo à la mesure.
  Détail : `packages/analyse/docs/fiche-modele.md` et `rapport-ablations.md`.
- Retour en arrière : `ALLIANCE_SCORE=regles` (déjà le défaut) ; tout échec de chargement ou de
  calcul retombe sur les règles avec un warning, le brief ne casse jamais.
