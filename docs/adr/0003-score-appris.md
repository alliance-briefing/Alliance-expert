# ADR-0003 : score de priorité appris (pipeline A) — caractéristiques lisibles et comparaisons de paires
Date : 2026-10-25, mise à jour le 2026-10-26 · Statut : proposé
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

### Choix du modèle d'embeddings
MiniLM multilingue (0,22 Go, déjà utilisé par le RAG) contre `multilingual-e5-large` (2,2 Go) : F1 « action » 0,69 contre 0,79, mais « réponse » 0,85 contre 0,84, « incident » 0,67 contre 0,63, et **même accord du score final sur les paires réservées : 0,86** (validation croisée sur l'entraînement : 0,81 contre 0,76). MiniLM est retenu par défaut (26/10) ; e5 reste disponible (`intents --e5`). Le poids de l'interaction « demande d'action du N+1 » change de signe d'un modèle à l'autre (+0,3 avec e5, −0,2 avec MiniLM, la hiérarchie seule passant à +0,75) : 12 paires ciblées ne suffisent pas à la fixer.

### Test de transfert : données de démo du lot 1
Autre métier (réseau d'expertise automobile), autre persona (Camille Laurent, responsable de l'agence Lyon-Est, N+1 Bernard Mercier), autres modèles de phrases, pas de corps complet (extrait seulement), pas de mails envoyés. 40 paires de 26 matins, jugées par Théo et **jamais apprises** : le modèle est celui entraîné sur les seules données de David.

| | Accord avec Théo (37 paires non équivalentes) |
|---|---|
| Score appris (entraîné sur David) | **0,89** (IC 95 % 0,78 – 0,97) |
| Score par règles | 0,59 (ne départage pas 24 paires sur 37) |
| Cohérence de Théo (4 répétitions côtés inversés) | 4/4 |

Écart +0,30 (IC 95 % +0,18 – +0,41). Paires au hasard : 0,84 contre 0,66 ; paires serrées : 0,94 contre 0,53. Les 4 erreurs : deux devis de garage, que Théo place avant un message d'expert (le modèle d'intentions, appris sur « Veuillez trouver ci-joint notre devis » chez David, n'y voit pas de demande d'action), une note d'information prise pour un incident, deux messages d'experts proches. Le score par règles n'a pas les tâches dans ces paires (comparaison de mails) : sur le brief réel, ses signaux de tâches restent utiles.

## Mise à jour du 26/10 : robustesse, données générées, annotation à deux, banc à l'aveugle
**Diagnostic.** Un banc de tests de comportement (CheckList, 61 vérifications sur des mails fictifs
indépendants, `scoring/ml/checklist.py`) montre que « réponse » est solide mais que « action » lit
le sujet du mail plutôt que l'acte : « je reviens vers vous » passait pour une demande,
« Merci de signer le bon de commande » non. Cause : 70 demandes d'action étiquetées, dans deux
styles seulement.

**Essais écartés, mesurés.**
- Indices d'actes de langage : +0,21 de F1 sur David, mais recul sur le banc et sur le
  transfert (le modèle avait appris que chez David les demandes sont des questions).
- SetFit (MiniLM réajusté par apprentissage contrastif) : 24 min d'entraînement sur processeur,
  F1 « action » 0,48 contre 0,69 sur la démo, rien de mieux sur le banc.

**Données générées** (`scoring/ml/synth/`, spécification `SPEC.md`) : mails isolés de cinq secteurs
(conseil, BTP, clinique, e-commerce, collectivité ; pas l'expertise automobile, gardée pour le
test), actes de langage × formulations × forme, étiquettes par construction. Puis des journées
de cinq managers sur trois semaines (affaires avec relance, réponse de la manager, clôture ;
automates, lettres d'information, invitations avec événement).

**Annotation à deux** (Théo et Paul, un fichier chacun, un quart en commun, désaccords arbitrés) :

| Tâche | Accord brut | Kappa de Cohen |
|---|---|---|
| Intention « action » (74 mails communs) | 95 % | 0,89 |
| Intention « réponse » | 91 % | 0,79 |
| Priorité des paires (47 communes) | 79 % (86 % hors « équivalent ») | 0,60 (0,72) |

Accord de chacun avec les étiquettes de construction : 90 à 95 %. Convention fixée à l'arbitrage :
une demande posée en question attend une réponse.

**Biais découvert : les pastilles.** Les paires de David et de la démo ont été jugées en voyant
les intentions calculées par un modèle (Théo suit la pastille « réponse » 95 % du temps). Elles
avantagent ce modèle et ne départagent pas deux modèles d'intentions. Nouveau juge : **80 paires
réservées des journées générées, jugées à l'aveugle** (fonction de l'expéditeur, ancienneté, fil,
réponse déjà envoyée ; aucune intention calculée), par deux annotateurs.

**Choix du modèle d'intentions** (règle fixée avant la mesure : le candidat est retenu s'il fait
au moins aussi bien que le modèle actuel sur les 80 réservées) :

| Intentions (régression logistique sur MiniLM) | 80 réservées à l'aveugle | David réservées | Démo |
|---|---|---|---|
| Entraînées sur David (181 mails) | 0,72 | 0,73 | 0,89 |
| **Entraînées sur 601 mails (David, démo hors paires, mails générés)** | **0,78** | **0,77** | 0,84 |
| Score par règles | 0,58 | 0,61 | 0,59 |

Le candidat est retenu (écart probable mais non démontré : environ 4 paires sur 74). Sur 70 mails
de démo étiquetés à la main, les mails générés font passer « action » de 0,62 à 0,69 de F1.

**Modèle final** (score appris sur 108 paires de David et 80 paires générées) : **0,80**
(IC 95 % 0,70 – 0,89) sur les 80 réservées à l'aveugle, contre 0,58 pour le score par règles,
qui ne départage pas 56 paires sur 74 ; 0,77 sur les réservées de David ; 0,78 sur la démo.
Plafond réaliste : l'accord entre les deux annotateurs, 86 % hors « équivalent ».

## Conséquences
- **Lot 2** : le score appris n'est pas encore branché dans le brief. Il reste à fixer le passage du score des mails au score de l'affaire (maximum des mails, bonus de taille du fil…) et à prévenir Marie-Yoram, qui branche le score.
- **Pipeline A** : l'étage « ajustement LLM ±15 » reste possible par-dessus, mais rien n'en dépend.
- **Production** : MiniLM (0,22 Go, fastembed ONNX, sans torch, déjà utilisé par le RAG) ; poids du score et des intentions versionnés (quelques Ko) ; vecteurs en cache hors du dépôt. Le groupe `nli` (torch) n'a servi qu'à l'essai du NLI.
- **Réentraîner** : toutes les étiquettes sont versionnées (`scoring/annotations/`, données synthétiques) ; `intents` puis `learned_score` les relisent et donnent les mêmes poids.
- **Retour arrière** : rien n'est branché ; le score par règles reste la référence.

## Limites
- **Poids instables pour les caractéristiques rares** (réunion dans les 48 h, échéance dans
  2 ou 3 jours, fil long) : leur signe change d'un entraînement à l'autre. « Pas encore répondu »
  reste négatif (−0,9) : à vérifier sur des données réelles avant de l'utiliser comme raison.
- **Penchant pour la gauche** chez un annotateur à l'aveugle (A choisi 64 fois sur 98) : neutralisé
  en moyenne par le tirage au hasard des côtés, mais source de bruit.
- **Données synthétiques à modèles de phrases** : un bon résultat ici ne prouve pas qu'on réussira sur de vrais mails. Le transfert vers la démo du lot 1 (autre métier, autre persona : 0,89) est encourageant, mais les deux jeux restent générés. Il faut refaire la mesure dès que le lot 1 fournit des données réelles.
- **Un seul annotateur** et 22 paires de mesure : intervalles de confiance larges. L'écart avec les règles est probable, son ampleur incertaine.
- **Hiérarchie** : sur les 120 premières paires, son poids était presque nul (+0,1) à intention égale ; la série ciblée l'a porté à +0,6, plus +0,3 pour l'interaction. Les mails du N+1 suivent ici les mêmes modèles de phrases que ceux des collègues : à revérifier sur des données réelles.
- **« Pas encore répondu » a un poids négatif** (−0,5), contre l'intuition : dans ces données, les mails sans réponse d'Hélène sont surtout des automates et des fournisseurs, et ceux auxquels elle a répondu des fils d'affaires actives. C'est une corrélation propre au jeu, pas une règle à garder telle quelle.
- **Caractéristiques rares** dans les paires (alerte, échéance proche, réunion dans les 48 h : aucune) : leurs poids sont peu sûrs.
- Les mots-clés des indices ont été écrits en connaissant le style des données.
