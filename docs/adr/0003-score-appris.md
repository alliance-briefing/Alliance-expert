# ADR-0003 : score de priorité appris (pipeline A) — caractéristiques lisibles et comparaisons de paires
Date : 2026-10-25 · Statut : proposé
Décideurs : lot 2 (Théo Szappanyos), à présenter à David

## Contexte
Réunion du 14/10 avec David : deux pipelines de scoring à comparer, A (règles → ML → ajustement LLM ±15) et B (LLM seul). Aucun classifieur de mails comparable à Jev (TypeSafe, hébergé, poids non publiés, français non annoncé) n'existe en français. David a confié à Théo la conception d'un modèle de ML **indépendant** (local, sans API, sans LLM en production), entraînable sans annotations du lot 3. L'API Claude a été écartée tant qu'aucun budget n'est accordé : elle est facturée hors abonnement.

Données : jeu Microsoft Graph synthétique de David (persona Hélène Marchand, directrice des opérations, 05/10 → 16/11 ; 4 463 mails dont 471 envoyés, 195 événements), converti en `Item` du contrat A (`scoring/ml/graph_convert.py`, code du lot 1 réutilisé en lecture seule). Mesuré sur ce jeu :
- `flag`, `categories` et `importance` sont tirés au hasard (2-3 %), `isRead` ne dépend que de l'âge, les réponses d'Hélène ne dépendent pas du contenu, elle est toujours en destinataire direct : **aucun signal de comportement n'est utilisable comme étiquette** ;
- le texte vient de modèles de phrases : 181 modèles une fois masqués nombres, jours, mois et parties variables. Toute mesure sépare donc entraînement et test par modèle de phrase ;
- pas de tâches : les échéances sont écrites dans le texte des mails.

## Options examinées
1. **Étiquettes faibles par les règles, puis ML** : le modèle apprend à copier les règles. Écarté.
2. **Étiquettes tirées du comportement (méthode Gmail Priority Inbox)** : impossible ici, le comportement est aléatoire dans les données.
3. **LLM professeur** (API ou local) : API sans budget ; un LLM local de 7-8 Md de paramètres sur CPU mettrait plus de 10 h pour 4 000 mails.
4. **NLI multilingue zéro-coup** (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, 83,4 % sur XNLI français) pour extraire les intentions : **testé et écarté**. F1 de 0,30 à 0,40 sur les intentions, parce qu'il lit les mots et pas l'intention (« merci de m'envoyer vos éléments » ≈ remerciement). Ses variantes (2,7 M de paires NLI, `bge-m3-zeroshot`) ne sont pas entraînées sur la pragmatique des mails.
5. **Étiquetage semi-manuel, retenu** : Théo étiquette des **caractéristiques** (oui/non, 2 s par mail) puis des **comparaisons de paires**, dans une page web locale au clavier. Le modèle apprend son jugement, et chaque poids est lisible.

## Décision
Un score en trois étages, tous locaux et déterministes :

1. **Règle de bruit** (`noise.py`) : un envoi en masse (lien de désabonnement) est écarté avant le modèle ; 1 084 mails sur 3 992, tous issus des 5 lettres d'information. « noreply » n'est **pas** du bruit : un rappel RH « noreply » demande une action. Le score par règles (`score.py`) distingue désormais « envoi en masse » (−30) et « notification automatique » (−10).
2. **Fiche lisible de chaque mail à 8 h 30** (`features.py`) :
   - expéditeur (hiérarchie, direction, collègue, client, fournisseur, automate ; configuration de persona) ;
   - intentions **apprises** sur 181 mails étiquetés par Théo (`intents.py` : `multilingual-e5-large` + régression logistique) : demande une action, attend une réponse, incident ;
   - intentions **par règle** quand le concept est précis : alerte de sécurité ou bancaire, échéance avec sa date (`deadline.py`) ;
   - fil sur 30 jours sans regarder le futur : réponse déjà envoyée, longueur, hiérarchie présente ;
   - fraîcheur, réunion dans les 48 h, interaction « demande d'action du N+1 » (critère de David : la hiérarchie pèse sur l'importance d'une action).
3. **Poids appris par Bradley-Terry** (`learned_score.py`) sur les comparaisons de paires de Théo : « ce matin, lequel traiter en premier ? ». C'est une régression logistique sur la différence de deux fiches. Score 0-100 (50 = mail médian) ; les raisons sont les plus fortes contributions (« attend une réponse », « échéance aujourd'hui ou demain »).

## Mesures
Intentions, sur des mails jamais vus (validation croisée imbriquée 5 plis × 5, un mail par modèle de phrase) :

| Caractéristique | Méthode | Précision | Rappel | F1 |
|---|---|---|---|---|
| demande une action | e5-large + LR | 0,74 | 0,85 | 0,79 |
| attend une réponse | e5-large + indice « ? » + LR | 0,85 | 0,82 | 0,84 |
| incident | e5-large + LR | 0,53 | 0,79 | 0,63 |
| échéance (+ date) | règle | 1,00 | 0,94 | 0,97 |
| alerte | règle | 1,00 | 1,00 | 1,00 (4 cas) |

Références : NLI zéro-coup 0,30 à 0,40 ; mots-clés seuls 0,42 à 0,59 (sauf « ? » pour « réponse », 0,85) ; MiniLM ou mpnet à la place d'e5 ≈ 0,72 sur « action ». Cohérence de Théo avec lui-même (10 % de répétitions cachées) : 94 à 100 %, 83 % pour « problème », trop flou, scindé en incident et alerte.

Score, sur 24 paires réservées (jamais vues par l'entraînement ; 22 non équivalentes) :

| | Accord avec Théo |
|---|---|
| Score appris | **0,86** (IC 95 % 0,73 – 1,00) |
| Score par règles (`score.py`) | 0,61 (il ne départage pas 15 paires sur 22) |
| Hasard | 0,50 |
| Cohérence de Théo (répétitions côtés inversés) | 0,92 |

Écart appris − règles : +0,25 (IC 95 % +0,09 – +0,41) : l'avantage est significatif. Validation croisée « une paire laissée de côté » sur les 108 paires d'entraînement : 0,76.

Série ciblée « N+1 » (12 paires) : une demande d'action du N+1 passe avant la demande d'action d'un autre expéditeur 6 fois sur 8, et avant un mail du N+1 sans demande 3 fois sur 4.

Poids principaux (modèle final, 132 paires) : attend une réponse +2,2 ; alerte +1,1 ; échéance aujourd'hui ou demain +1,1 ; demande une action +1,0 ; échéance plus tard +0,7 ; hiérarchie +0,6 et demande d'action du N+1 +0,3 ; fil long +0,4 ; automate −0,4 ; « pas encore répondu » −0,5 ; fournisseur −0,9.

## Conséquences
- **Lot 2** : le score appris n'est pas encore branché dans le brief. Il reste à fixer le passage du score des mails au score de l'affaire (maximum des mails, bonus de taille du fil…) et à prévenir Marie-Yoram, qui branche le score.
- **Pipeline A** : l'étage « ajustement LLM ±15 » reste possible par-dessus, mais rien n'en dépend.
- **Production** : e5-large (2,2 Go, fastembed ONNX, sans torch) ; poids du score et des intentions versionnés (quelques Ko) ; vecteurs en cache hors du dépôt. Le groupe `nli` (torch) n'a servi qu'à l'essai du NLI.
- **Réentraîner** : toutes les étiquettes sont versionnées (`scoring/annotations/`, données synthétiques) ; `intents` puis `learned_score` les relisent et donnent les mêmes poids.
- **Retour arrière** : rien n'est branché ; le score par règles reste la référence.

## Limites
- **Données synthétiques à modèles de phrases** : un bon résultat ici ne prouve pas qu'on réussira sur de vrais mails. Il faut refaire la mesure dès que le lot 1 fournit des données réelles ou plus variées.
- **Un seul annotateur** et 22 paires de mesure : intervalles de confiance larges. L'écart avec les règles est probable, son ampleur incertaine.
- **Hiérarchie** : sur les 120 premières paires, son poids était presque nul (+0,1) à intention égale ; la série ciblée l'a porté à +0,6, plus +0,3 pour l'interaction. Les mails du N+1 suivent ici les mêmes modèles de phrases que ceux des collègues : à revérifier sur des données réelles.
- **« Pas encore répondu » a un poids négatif** (−0,5), contre l'intuition : dans ces données, les mails sans réponse d'Hélène sont surtout des automates et des fournisseurs, et ceux auxquels elle a répondu des fils d'affaires actives. C'est une corrélation propre au jeu, pas une règle à garder telle quelle.
- **Caractéristiques rares** dans les paires (alerte, échéance proche, réunion dans les 48 h : aucune) : leurs poids sont peu sûrs.
- Les mots-clés des indices ont été écrits en connaissant le style des données.
