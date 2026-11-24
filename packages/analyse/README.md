# packages/analyse — lot 2 (Analyse)

Transforme les `Item` d'un utilisateur (contrat A) en `Brief` (contrat B).

## État actuel : brief à base de règles + score de priorité (L2.0–L2.2), par affaires (L2.1)

```python
from contracts.demo_data import load_items
from packages.analyse import generate_brief

brief = generate_brief(load_items("2026-10-07"), "u_demo_001", "2026-10-07")
```

| Section | Règle de sélection | Une entrée = | Ordre | `priority_score` |
|---|---|---|---|---|
| `actions` | affaires sans réunion du jour : mail non lu **ou** score > 50 ; 7 au plus | les mails non lus d'une affaire (sinon le mail qui porte le score appris, sinon le premier élément ; le texte ne dit « non lu » que s'il l'est) | score décroissant, ex aequo par `case_id` | score appris, sinon `score_case` |
| `meetings` | événements non annulés qui occupent au moins une partie du jour ; une affaire déjà dans `meetings` ne crée **aucune** entrée dans les autres sections | une réunion (une par heure, jamais fusionnée) | heure de début | 50 fixe (ordre par heure, pas par score) |
| `overdue` | affaires sans réunion du jour et avec au moins une tâche en retard | les tâches en retard d'une affaire | score décroissant, ex aequo par `case_id` | score appris, sinon `score_case` |

**Raisons** (`priority_reasons`, max 4) : jusqu'à 3 raisons de `score_case` (sans « élément
professionnel standard »), puis la raison de contexte de l'affaire si elle a d'autres éléments
(« dossier 2026-3025 : aussi 1 mail, 1 tâche en retard »). Repli sur la raison de rédaction
quand le score est neutre et qu'il n'y a pas de contexte.

**Réunions annulées** (`attributes.calendar.is_cancelled`) : elles ne génèrent pas d'entrée dans
`meetings`, mais si une réunion annulée appartient à une affaire, elle est annoncée dans la raison
de contexte (« 1 réunion annulée ») et citée dans `source_item_ids`.

Toutes les réunions du jour d'une affaire restent dans `meetings` ; leurs autres éléments
(mails non lus, tâches en retard…) sont résumés dans la raison de contexte de chaque entrée.
Exception : deux réunions de la même affaire le même jour → deux entrées (une par heure).

- Isolation : seuls les items dont `owner_user_id` est l'utilisateur demandé sont lus ; les autres sont comptés dans `warnings`.
- Déterminisme : même entrée, même brief, quel que soit l'ordre des items (hors `generated_at` et `stats.latency_ms`). « Maintenant » = `collected_at` le plus récent, jamais `datetime.now()`.
- Les textes ne reprennent que les titres et les dates, jamais le contenu des mails.
- Claims (L2.4) : le texte de chaque entrée est découpé en faits, un par élément montré
  (« Mail non lu « Devis », reçu le 07/10 à 08:15 »), cité par ce seul élément ; au-delà de 3,
  un claim compte le reste (« Et 3 autres mails non lus ») et le cite. Le contexte de l'affaire
  reste dans les raisons.
- Garde (L2.4, `guard/check.py`), porte dure appliquée à la fin de `generate_brief`, sans LLM :
  un claim n'est livré que si ses items cités existent et appartiennent à l'utilisateur, et si
  son titre entre « », ses dates (avec l'heure, en heure de Paris), nombres, noms propres, type
  et état (non lu, annulée, en retard) se retrouvent dans ces items ; « Et N autres… » doit citer
  N items. Sinon l'entrée est dégradée (« À consulter : « titre » », lien, sans claim) ; sans
  source valide, elle est retirée et les rangs sont renumérotés. Chaque cas ajoute un warning.
  Démo : 424 claims sur 30 journées, aucun rejeté. 20 cas adverses (date ou heure fausse, heure
  UTC non convertie, montant inversé, numéro voisin, homonyme, item d'un autre utilisateur…),
  tous rejetés : `tests/test_guard_adversarial.py`.
- Trace d'une entrée : chaque claim, son verdict par la garde, ses items sources (lien, dates en
  heure de Paris, celles citées marquées « (citée) », extrait) et les items de contexte. Le brief
  est régénéré (déterministe) à partir des items de la journée, démo ou `--dossier <DATA_DIR>`.

```bash
uv run python -m packages.analyse.guard.trace b_2026-10-07_u_demo_001 1                      # rang 1 de chaque section
uv run python -m packages.analyse.guard.trace b_2026-10-07_u_demo_001 2 --section meetings
```
- Heure de Paris (`clock.py`) : toutes les dates sont lues en Europe/Paris quel que soit le décalage
  reçu (un item en UTC venant de Graph est donc bien placé), `generated_at` compris. Exception :
  une journée entière, dont les dates se lisent telles quelles. Elle est donnée par
  `attributes.calendar.is_all_day` (contrat A 1.1) ; sans ce bloc (items 1.0), on la déduit :
  minuit à minuit dans son propre décalage.
- Score (L2.2) : **score appris** (`scoring/learned.py`) pour les affaires qui ont un mail à
  noter (reçu, hors envoi en masse, non lu ou reçu dans les 3 jours) : l'affaire prend le score
  de son meilleur mail, plus les points des règles pour ce que le modèle ne voit pas (tâche en
  retard +25 ou à échéance +20 / +10, réunion dans les 48 h +15). Une intention apprise n'est
  une raison que si sa probabilité atteint 0,5. Les autres affaires gardent `score_case`
  (`scoring/score.py`) ; réunions : score fixe 50 (ordre par heure).
- Persona (qui est le N+1, qui est client) : configuré pour `u_demo_001` et `u_graph_helene` ;
  sinon adresse du manager déduite (destinataire principal le plus fréquent), sans hiérarchie.
- `ALLIANCE_SCORE=regles` : score par règles seul (comparaison A/B). Si le modèle manque ou
  échoue, le brief garde les règles et l'annonce dans `warnings`. Les tests tournent par
  défaut avec les règles (`conftest.py` à la racine) ; ceux du score appris passent un faux modèle.
- Coût mesuré (démo du 07/10, CPU) : 3,3 s pour le premier brief (chargement du modèle), puis
  1,7 s. L'image Docker embarque le modèle (`FASTEMBED_CACHE_PATH=/app/.models`).
- `generator.llm` : `"MiniLM (score de priorité appris)"` (modèle dans `llm_version`) quand le
  score appris a noté au moins une affaire, sinon `"rules"`. Aucun LLM, aucun coût d'API.

## Agrégation en affaires (L2.1, niveaux 1 et 2)

```python
from packages.analyse.aggregate.cases import group_into_cases

cases = group_into_cases(items)  # list[Case] : case_id, items, links
```

Deux items sont dans la même affaire s'ils partagent un lien ; les liens sont transitifs.

| Niveau | Lien | Exemple de `links` |
|---|---|---|
| 1 | même `thread_id` | `fil:conv:dm1-conv-000024` |
| 1 | même `correlation_id` (contrat A 1.1 : invitation reçue par mail + événement, rempli par le lot 1) | `réunion:meet:point-flash:2026-10-21T07:00Z` |
| 2 | même référence dans le **titre** | `dossier:2026-3025` |
| 2 | même document : le fichier et les titres qui le citent entre « » (nom avec extension) | `document:Planning_experts_S41.xlsx` |

Le type d'une référence dépend du mot qui précède le numéro (`REFERENCE_KINDS`) :
dossier, mission, expertise, rapport, devis → `dossier` ; facture → `facture` ; commande → `commande` ;
tout autre mot ne crée aucun lien. Une facture et une commande au même numéro restent donc séparées.
Les « _ » comptent comme des espaces : `Rapport_expertise_2026-2367.docx` rejoint le dossier 2026-2367.
Un texte entre guillemets sans extension (« Expertises complexes ») est un canal de discussion :
il ne crée aucun lien.

- Titre seulement : un extrait cite souvent d'autres dossiers et fusionnerait des affaires sans rapport.
- Isolation : deux utilisateurs ne partagent jamais une affaire.
- Déterminisme : `case_id = "case:" + plus petit item_id` ; affaires triées par `case_id`,
  items par date de création.
- Le niveau 3 (sujet proche, embeddings) viendra si la mesure du lot 3 montre un rappel
  insuffisant, derrière un veto « références différentes » : l'exploration (`rag/explore.py`)
  montre que les messages-types (« Dossier 2026-5394 : RAS » / « Dossier 2026-4752 : RAS »)
  sont presque identiques pour le modèle.

## Embeddings (L2.6, en cours)

`rag/embeddings.py` : vecteurs de sens des items, base du futur niveau 3 de L2.1 et du RAG.
Pas encore utilisé par le brief.

```python
from packages.analyse.rag.embeddings import FastEmbedder, embed_items, similarities

vectors = embed_items(items, FastEmbedder())  # {item_id: vecteur normalisé}
```

- `Embedder` remplaçable : `FastEmbedder` (fastembed, `paraphrase-multilingual-MiniLM-L12-v2`,
  384 dimensions, 0,22 Go, local, sans PyTorch) ou `HashEmbedder` (faux modèle des tests).
- Texte vectorisé (`item_text`) : titre sans « RE: », « TR : »…, puis extrait.
- Cache par (modèle, propriétaire, item_id, updated_at) : un item n'est jamais recalculé.
- fastembed est une dépendance principale depuis le branchement du score appris (le groupe
  `embeddings` reste pour les anciennes commandes). Le modèle est téléchargé au premier appel,
  ou à la construction de l'image Docker.
- Mesure locale (CPU) : chargement 2,6 s, puis 500 textes en 1,4 s ; similarité d'une paraphrase
  française 0,71 contre 0,16 pour un texte sans rapport.

### Index par utilisateur (`rag/index.py`, étape R1)

```python
from packages.analyse.rag.embeddings import FastEmbedder
from packages.analyse.rag.index import IndexStore

store = IndexStore(FastEmbedder())
index = store.for_user("u_demo_001")
index.add(items_des_jours_passes)
hits = index.search_items(items_du_jour, date_du_brief, exclude=ids_du_jour)  # k=3, 30 jours
store.delete_user("u_demo_001")  # suppression complète
```

- Isolation : un index par utilisateur ; il refuse les items et les requêtes d'un autre.
- Pas de fuite du futur : seuls les items créés avant le jour du brief (heure de Paris), dans
  les 30 jours précédents, sont renvoyés.
- Une version plus récente d'un item remplace l'ancienne ; tri déterministe (score, puis item_id).
- En mémoire (NumPy) ; le stockage persistant sera tranché dans l'ADR RAG. Pas encore branché
  dans le brief (étapes R2-R3).

### Historique des affaires (`rag/retrieve.py`, étape R2)

`history(cases, index, jour)` → pour chaque affaire du jour, au plus 3 items passés liés
(`PastLink` : item, score, raison). Rien n'est encore affiché dans le brief (R3).

- Requête : chaque item de l'affaire interroge l'index ; un item passé garde sa meilleure similarité.
- Même référence (« même dossier 2026-3025 ») : lien certain, accepté quel que soit le score.
- Sinon similarité ≥ seuil, sauf veto si l'affaire et l'item citent des références toutes
  différentes. Seuil provisoire 0,85 (prudent), à régler avec `explore --historique` (R2b).
- Liens certains d'abord, puis score, puis item_id ; items du jour et de l'affaire exclus.

### Exploration (`rag/explore.py`)

Ce que trouverait un niveau 3 de L2.1 : distribution des similarités (paires déjà reliées par les
niveaux 1-2 contre les autres) et paires les plus proches non reliées, à juger à la main.

```bash
uv run python -m packages.analyse.rag.explore 2026-10-07                 # titre + extrait
uv run python -m packages.analyse.rag.explore 2026-10-07 --texte titre --top 40
uv run python -m packages.analyse.rag.explore 2026-10-07 --modele hash   # faux modèle, instantané
uv run --with model2vec python -m packages.analyse.rag.explore 2026-10-07 --modele model2vec
uv run python -m packages.analyse.rag.explore <jour> --dossier <DATA_DIR> --utilisateur <id>
```

Historique des affaires (R2b) : l'index reçoit les N journées précédentes, puis chaque affaire
du jour affiche ses items passés retenus et écartés, avec la raison (`retrieve.verdict`).

```bash
uv run python -m packages.analyse.rag.explore 2026-10-09 --historique --jours-historique 5 --seuil 0.85
```

Résultats (démo, MiniLM-L12, 5 jours d'historique, 09/10, 15/10 et 21/10, jugés à la main) :
les liens de même référence sont justes ; les rattachements « sujet proche » sont faux dans
toutes les tranches de score jugées (0,75 à 1,00) : noms de fichiers générés
(`Compte_rendu_20261007_01.docx` ← `Compte_rendu_20261006_52.docx`, photos `IMG_…`),
occurrences d'une même réunion récurrente, messages-types (« Déjeuner – Maëlle » ←
« Déjeuner – Jérôme »). Les scores les plus hauts sont les plus trompeurs. Réserve : données de
démo générées à partir de modèles de phrases. Le seuil par défaut (0,85) est en attente de
décision ; piste : similarité désactivée par défaut, ou encadrée par des garde-fous
(participant commun, fichiers et séries exclus).

## Score appris (L2.2, pipeline A, en cours)

Score de priorité appris à partir de caractéristiques lisibles (signaux des règles +
intentions des mails) et de comparaisons de profils annotées à la main. Pas encore utilisé
par le brief.

**Données** : `scoring/ml/graph_convert.py` convertit le jeu Microsoft Graph synthétique de
David (`alliance-graph-synth`, persona Hélène Marchand, 05/10 → 16/11) en Item du contrat A,
utilisateur `u_graph_helene`. Le texte, les invitations et les événements passent par le code
du lot 1 (`packages/collect`, importé en lecture seule). Ce que le contrat n'a pas (place du
manager en `to`/`cc`, mail envoyé…) va dans `MailExtras`, à côté de l'Item.

```bash
uv run python -m packages.analyse.scoring.ml.graph_convert <alliance-graph-synth>/data
# Item : 4658 {'mail': 4463, 'calendar': 195} (dont 471 mails envoyés), refusés : 0
```

Limites du jeu, mesurées : `flag`, `categories` et `importance` sont tirés au hasard (2-3 %),
`isRead` ne dépend que de l'âge, les réponses d'Hélène ne dépendent pas du contenu, et elle est
toujours en `to` : aucun de ces champs ne sert d'étiquette. Le texte vient de modèles de phrases
(≈ 400 débuts de corps) : séparer entraînement et test par modèle de phrase, pas au hasard.

**Bruit certain** : `scoring/ml/noise.py` écarte avant le modèle les envois en masse, repérés
par un lien de désabonnement dans l'objet ou le corps. Ni l'adresse (« noreply ») ni le mot
« newsletter » ne suffisent : un rappel RH « noreply » demande une action. Sur le jeu de David :
1 084 mails écartés sur 3 992 reçus, tous issus des 5 lettres d'information ; portail RH,
tickets, banque, sécurité et catalogues restent notés par le modèle.

```bash
uv run python -m packages.analyse.scoring.ml.noise <alliance-graph-synth>/data  # compte + échantillon
```

**Intentions des mails** (`scoring/ml/intents.py`), étiquetées par Théo sur 181 mails (un par
modèle de phrase, hors bruit : `scoring/ml/sample.py`) avec la page locale
`uv run python -m packages.analyse.scoring.ml.annotate` (http://127.0.0.1:8765). Réponses dans
`scoring/annotations/intent_labels.jsonl` (ajout seul ; 10 % de répétitions cachées).
Cohérence de l'annotateur : 94 à 100 % selon la question, 83 % pour « problème », trop flou,
scindé en « incident » et « alerte » (revue de 53 mails, les autres héritent du « non »).

Mesure sur des mails jamais vus (validation croisée imbriquée, 5 plis × 5) :

| Caractéristique | Méthode | Précision | Rappel | F1 |
|---|---|---|---|---|
| action | e5-large + régression logistique | 0,74 | 0,85 | 0,79 |
| reponse | e5-large + indice « ? » + régression logistique | 0,85 | 0,82 | 0,84 |
| incident | e5-large + régression logistique | 0,53 | 0,79 | 0,63 |
| échéance (+ date) | règle `scoring/ml/deadline.py` | 1,00 | 0,94 | 0,97 |
| alerte | règle (automate + « inhabituel », « connexion »…) | 1,00 | 1,00 | 1,00 (4 cas) |

Écartés, mesurés : NLI zéro-coup `mDeBERTa-v3-base-mnli-xnli` (F1 0,30 à 0,40 : il lit les mots,
pas l'intention, « merci de m'envoyer » ≈ remerciement), mots-clés seuls (0,42 à 0,59 sauf
« ? » pour reponse, 0,85), MiniLM et mpnet (≈ 0,72 sur action). Quand le concept est précis
(échéance, alerte), la règle bat le modèle ; flou mais fréquent (action, réponse), le modèle
appris bat les règles. Réserve : données synthétiques, mots-clés écrits en connaissant leur style.

- `uv run --group embeddings python -m packages.analyse.scoring.ml.intents` : mesure puis
  entraîne et écrit `scoring/ml/intents_model.npz` (poids, quelques Ko ; déterministe).
- MiniLM (0,22 Go) ou e5-large (2,2 Go), fastembed ONNX, sans torch ; vecteurs en cache par
  (modèle, texte), un fichier par modèle, dans `~/.cache/alliance-briefing/`, hors du dépôt.
- Le groupe `nli` (torch, transformers) n'a servi qu'à l'essai du NLI. Piège noté : transformers 5
  garde le float16 du fichier, 11 fois plus lent sur CPU : charger en `dtype=torch.float32`.
- Le test sur le jeu complet de David tourne si `ALLIANCE_GRAPH_DIR` pointe vers son `data/`.

**Score appris** (`scoring/ml/learned_score.py`, décision et limites : `docs/adr/0003-score-appris.md`).
Fiche de chaque mail à 8 h 30 (`scoring/ml/features.py` : expéditeur, intentions, échéance, fil
sur 30 jours, fraîcheur, demande d'action du N+1), poids appris par Bradley-Terry sur 132
comparaisons de paires de Théo (`scoring/ml/pairs.py`, page `/paires` de l'annotation).
Score 0-100 (50 = mail médian) et jusqu'à 3 raisons. Sur 24 paires réservées : accord avec Théo
0,86 (IC 95 % 0,73-1,00) contre 0,61 pour le score par règles, qui ne départage pas 15 paires
sur 22 ; cohérence de Théo 0,92.
Transfert vers la démo du lot 1 (autre métier, persona Camille Laurent, 40 paires jamais apprises) :
0,89 (IC 95 % 0,78-0,97) contre 0,59 pour les règles. Embeddings : MiniLM par défaut (même score
final qu'e5-large, 10 fois plus léger) ; `intents --e5` pour e5.
**Depuis le 26/10** (ADR-0003, mise à jour) : intentions entraînées sur 601 mails (David, démo hors
paires, mails générés annotés à deux) ; score appris sur 108 paires de David et 80 paires de
journées générées. Mesure principale : **0,80** (IC 95 % 0,70-0,89) sur les 80 paires réservées
des journées générées, jugées à l'aveugle par deux annotateurs, contre 0,58 pour le score par
règles ; plafond réaliste : l'accord entre annotateurs, 86 % hors « équivalent ».

```bash
uv run --group embeddings python -m packages.analyse.scoring.ml.pairs <data>   # paires
uv run python -m packages.analyse.scoring.ml.learned_score                      # mesure + poids
```

### Robustesse des intentions et données générées (en cours)
- **Banc CheckList** (`scoring/ml/checklist.py`, `annotations/intent_checklist.json`) : 61
  vérifications sur des mails fictifs indépendants (cas de base, invariance, direction).
  « réponse » réussit tout ; « action » échoue sur des demandes explicites ou implicites : le
  modèle lit le sujet du mail plutôt que l'acte. Deux essais (indices d'actes de langage, 2e série
  d'étiquettes) ont gagné sur David et reculé sur le banc ou le transfert : écartés, la règle étant
  de ne retenir un modèle que s'il ne recule sur aucun des trois garde-fous (banc, paires
  réservées de David, transfert vers la démo).
- **2e série d'étiquettes** (`scoring/ml/sample_round2.py`) : 120 mails, démo hors paires de démo
  et mails de David où le modèle hésite ; gardée pour la suite.
- **Générateur du lot 2** (`scoring/ml/synth/`, spécification `SPEC.md`) : mails isolés de cinq
  secteurs (conseil, BTP, clinique, e-commerce, collectivité ; pas l'expertise automobile, gardée
  pour le test), actes de langage × formulations × forme, étiquettes par construction jamais
  montrées aux annotateurs.

```bash
uv run python -m packages.analyse.scoring.ml.synth.generator        # 300 mails générés
uv run python -m packages.analyse.scoring.ml.annotate --annotateur theo   # ou paul
```

Annotation à deux : un fichier par personne (`intent_labels_theo.jsonl`, `intent_labels_paul.jsonl`),
un mail sur quatre en commun pour l'accord entre annotateurs, le reste partagé. Accord sur 74 mails
communs : action 95 % (kappa 0,89), réponse 91 % (kappa 0,79) ; désaccords arbitrés
(`intent_labels_arbitrage.jsonl`). Sur 70 mails de démo étiquetés à la main, les mails générés
font passer « action » de 0,62 à 0,69 de F1.

**Journées générées** (`scoring/ml/synth/days.py`) : cinq personas sur trois semaines, affaires
qui vivent (relance, réponse de la manager, clôture), flux de fond, automates, lettres
d'information, invitations avec leur événement. 160 paires du même matin, dont 80 réservées à la
mesure, annotées **à l'aveugle** (fonction de l'expéditeur et faits visibles, aucune intention
calculée : les paires jugées avec les pastilles d'un modèle avantagent ce modèle) et à deux.

```bash
uv run --group embeddings python -m packages.analyse.scoring.ml.synth.days     # journées + paires
uv run python -m packages.analyse.scoring.ml.annotate --annotateur theo --paires
```

## Outil d'annotation unifié (M4)

`scoring/ml/annoter.py` : une page locale pour les trois jugements du plan (élément : action,
urgence, alerte ; paire d'éléments ; paire d'affaires), toutes sources. Il lit une **série**
(`scoring/annotations/series/<série>.jsonl`) préparée par le tirage des jeux gelés (M5) ou par
l'apprentissage actif (M7) ; il n'échantillonne rien lui-même. Les cartes ne montrent que ce
qu'un connecteur expose (contrat A) : une série qui contient une étiquette ou un score est
refusée. Un fichier de réponses par annotateur, en ajout seul ; 10 % de répétitions cachées
(paires côtés inversés) ; kappa, puis arbitrage des désaccords. Les affaires sont les affaires
vraies du simulateur.

```bash
uv run python -m packages.analyse.scoring.ml.annoter --creer-essai                 # série d'essai
uv run python -m packages.analyse.scoring.ml.annoter --serie essai --annotateur theo
uv run python -m packages.analyse.scoring.ml.annoter --accord essai               # kappa, cohérence
uv run python -m packages.analyse.scoring.ml.annoter --serie essai --arbitrage
```

## Jeux de mesure gelés (M5) et compréhension des éléments (M6)

`scoring/ml/jeux_m5.py` tire et gèle les jeux de mesure, avant tout réglage : T-gen (150 éléments,
150 paires, 50 paires d'affaires sur les 4 métiers réservés) et T-démo (40 paires de la démo du
lot 1). `scoring/ml/comprendre.py` ajoute une série de 60 éléments de démo (`m6-tdemo-elements`)
et la **référence de M6** : MiniLM gelé + régression logistique (numpy), qui prédit l'action,
l'urgence et l'alerte des 4 sources à partir de la carte que voit l'annotateur. Elle apprend sur
les personas « train » du simulateur (gabarits + LLM), se règle sur 7 personas de validation et
n'est mesurée sur les jeux gelés que par `--mesurer`, qui refuse un gel manquant ou modifié et
inscrit chaque lecture au journal (`scoring/annotations/journal-geles.jsonl`). Le journal
identifie le modèle mesuré par son empreinte SHA-256, stable ; ses commits sont ceux de la
branche avant son rebasage puis sa fusion en un seul commit (857b546 et 81f2b25, devenus 3910cea
et 08619d9 sur `l2/feat/m6-reference`, contenu de `packages/` identique). Jugés par Théo
seul (décision du 21/11), sans proposition d'un modèle.

| F1 macro de l'action | validation | T-gen | démo |
|---|---|---|---|
| référence, étiquettes du 22/11 | 0,887 | 0,651 [0,565 ; 0,723] | 0,175 [0,096 ; 0,251] |
| référence, règles du guide du 23/11 | 0,868 | 0,625 [0,545 ; 0,695] | 0,164 [0,098 ; 0,230] |

Résultat négatif (G6) : la référence apprend la formulation du simulateur et ne transfère pas
vers la démo, et les règles de frontière du 23/11 n'y changent rien (égalité au sens de G2). La
référence n'est pas branchée dans le brief ; le réajustement de l'encodeur est reporté en bonus.

**Validation dure** (`--valider-dur`, étape 0 A du plan d'amélioration). La validation de 7
personas, une par métier, est trop optimiste : elle donne 0,868 quand T-gen donne 0,625. Pour
comparer les leviers suivants sans relire les jeux gelés (P3), trois découpages des personas
« train », tous sur des métiers jamais vus :

1. **métiers** : 4 plis de 3 métiers entiers sur 12 ; chaque carte est mesurée une fois ;
2. **gabarits** : en plus, une part des variantes de chaque famille de gabarits est retirée de
   l'entraînement (une variante au moins reste par famille) ; mesure sur les cartes qui les
   utilisent, comparée au découpage 1 sur les mêmes cartes ;
3. **écrivain** : sur les mails qui ont les deux textes, appris sur un écrivain, mesuré sur
   chacun (mêmes mails dans les quatre cases).

Le réglage est choisi par découpage selon G2 puis G3 : le plus simple dont l'écart au meilleur
n'est pas significatif. Cette correction vaut aussi pour `--entrainer` ; la référence commitée,
choisie avant au meilleur score strict, n'est pas réentraînée (sinon il faudrait la remesurer).
Rapport : `scoring/ml/comprendre_validation_dure.json`.

| F1 macro de l'action (référence MiniLM gelé + régression) | valeur [IC 95 %] |
|---|---|
| validation de 7 personas (rapport de `--entrainer`) | 0,868 |
| 1. métiers tenus à l'écart (23 390 cartes) | 0,728 [0,721 ; 0,735] |
| 2. gabarits tenus à l'écart (3 108 cartes) | 0,578 [0,555 ; 0,598], −0,166 sur les mêmes cartes |
| 3. écrivain : gabarits → gabarits / → LLM (6 409 mails) | 0,751 / 0,501 |
| 3. écrivain : LLM → LLM / → gabarits | 0,674 / 0,524 |
| pour mémoire : T-gen, 2e mesure | 0,625 [0,545 ; 0,695] |

Lecture : tenir des métiers entiers à l'écart retire une partie de l'optimisme (0,868 →
0,728), pas tout. Le découpage 1 reste au-dessus de l'intervalle de T-gen. Explications
possibles, non vérifiées : T-gen a une moitié de non-mails (75/25/25/25), où la référence est
faible (agenda 0,48, fichiers 0,46), et il est jugé par Théo, pas par l'étiqueteur du simulateur.
Les découpages 2 et 3 confirment le diagnostic de M6 : une formulation jamais vue coûte 0,17,
un changement d'écrivain 0,17 à 0,23. Le découpage 1 sert à suivre le niveau, les découpages 2
et 3 à mesurer ce que les leviers apportent contre l'apprentissage du style.

**Jeu de développement réel** (étape 0 B ; règle M0 modifiée le 22/11, plan §11). 104 cartes
de texte écrit hors de nos générateurs : les 34 éléments de la journée de Claire (écrits à la
main par Paul) et 70 éléments du jeu de David (55 mails reçus, 15 réunions), un par modèle de
phrase. Ces modèles sont inscrits au tirage (`m6-dev-reel.tirage.json`) et seront exclus du
futur test de transfert sur David. Claude a jugé les 104 cartes avec le guide avant toute
mesure ; Théo rejuge 30 cartes à l'aveugle (`m6-dev-reel-echantillon`). Sous 85 % d'accord
sur l'action, il arbitre les désaccords de l'échantillon (`m6-dev-reel_arbitrage.jsonl`).
Les étiquettes sont gelées (`m6-dev-reel.gel.json`) : `--valider-reel` refuse de mesurer si
elles ont changé. Jugé surtout par un LLM : c'est un garde-fou de développement, pas une
mesure humaine (P2).

Accord de Théo avec Claude sur l'échantillon : action 25/30 (83 %, sous le seuil), urgence et
alerte 30/30 ; 3 répétitions cachées cohérentes. Arbitrage de Théo : les 5 désaccords gardent
l'étiquette de Claude (4 mails qui annoncent une réunion sans invitation visible, règle ajoutée
au guide ; une tâche « Répondre à… », règle du verbe de l'intitulé). Une carte de l'échantillon
a été jugée après que Théo a lu la convention de Claude sur ces mails : non aveugle sur ce point.

| référence de M6 (MiniLM gelé + régression), jeu réel | F1 macro [IC 95 %] |
|---|---|
| action, 104 cartes | 0,229 [0,155 ; 0,304] (Claire 0,273, David 0,197) |
| urgence | 0,358 [0,298 ; 0,470] |
| alerte | 0,495 ; la seule alerte (nouvel appareil connecté) est manquée |

Le jeu réel retrouve l'échec de transfert vu sur la démo (0,164), quand la validation dure suit
plutôt T-gen : les deux parties du jeu de développement se complètent. Erreur la plus nette :
les envois en masse de David (Rien) sont lus comme Faire ou Relire.

**Levier 1, fiche hybride** (`--levier-fiche`, règle de décision écrite le 22/11 avant la mesure,
plan §11). La fiche (39 valeurs) est lue sur la carte, comme l'annotateur la voit : source, place
du manager, relation de l'expéditeur, mail envoyé par lui, réponse, transfert, importance, non lu,
lien de désabonnement ; réponse à l'invitation, organisateur, annulée, jour ; échéance, retard,
terminée, Planner, verbe de l'intitulé d'une tâche ; fichier modifié par lui, cité dans. Trois
variantes, toutes réentraînées de la même façon (réglage par G2 puis G3 sur le découpage 1) :

| F1 macro de l'action | R, vecteur | H, vecteur + fiche | F, fiche seule |
|---|---|---|---|
| 1. métiers | 0,728 | **0,804** | 0,542 |
| 2. gabarits tenus à l'écart | 0,578 | **0,683** | 0,513 |
| 3. gabarits → LLM / LLM → gabarits | 0,501 / 0,524 | **0,571 / 0,620** | 0,428 / 0,435 |
| jeu réel (104 cartes) | 0,220 | 0,382 | **0,495** |
| alerte réelle détectée | 0/1 | 0/1 | 1/1 |
| précision des alertes au rappel 0,95 (découpage 1) | 0,857 | 0,911 | 0,056 |

Écarts appariés H − R : +0,076 (métiers), +0,105 (gabarits), +0,070 et +0,096 (écrivain),
+0,162 [0,082 ; 0,231] (jeu réel), tous significatifs. **Décision de la règle : H est retenue**
et devient le candidat de la compréhension. F − H est significatif dans les deux sens (−0,26 sur
le simulateur, +0,113 [0,026 ; 0,212] sur le jeu réel) : F ne remplace pas H. Lecture : la fiche
porte l'essentiel du transfert ; sur le texte réel, le vecteur MiniLM gêne encore, car il reste
appris sur le style du simulateur. C'est la cible des leviers suivants (augmentation du texte,
encodeur réajusté), à mesurer contre H. R est ici réentraînée sur tout l'entraînement : 0,220 sur
le jeu réel, contre 0,229 pour la référence commitée (autre réglage, sans la validation).

**Levier 2, augmentation du texte** (`--levier-texte`, règles écrites le 23/11 avant la mesure,
plan §11). Toutes les variantes ont la fiche ; grille réduite à l2 ∈ {1e-2, 1e-3, 1e-4} pour
toutes, H comprise (d'où 0,596 au lieu de 0,620 pour H en LLM → gabarits). N : l'encodeur lit
l'objet et le texte nettoyé (sans date, champs, bloc Teams, désabonnement ni signature ;
chiffres, dates, liens et adresses masqués). A : une copie perturbée de chaque carte à
l'entraînement. Un groupe doit gagner avec chacune de ses 3 graines : une graine 0 perdante
arrête son groupe (même décision, moins de calcul).

| F1 macro de l'action | H | **N, normalisé** | A0, copies | N+A0 |
|---|---|---|---|---|
| métiers | 0,804 | **0,851** | 0,810 | 0,857 |
| gabarits tenus à l'écart | 0,683 | **0,716** | 0,681 | 0,726 |
| gabarits → LLM / LLM → gabarits | 0,571 / 0,596 | **0,634 / 0,679** | 0,575 / 0,605 | 0,630 / 0,684 |
| jeu réel | 0,382 | **0,445** | 0,445 | 0,440 |
| précision des alertes au rappel 0,95 | 0,911 | **0,998** | 0,870 | 0,995 |

**Décision de la règle : N est retenue.** N − H est significatif sur les découpages 2 et 3
(+0,033 à +0,084) et sur le découpage 1 (+0,048) ; sur le jeu réel, +0,062 [−0,022 ; 0,147],
dans le bruit. A0 (gabarits et réel meilleurs que H) ne bat pas N : A − N est significatif à la
baisse sur la validation dure, nul sur le jeu réel. N+A0 − N n'est significatif que sur le
découpage 1 (+0,006), pas sur une mesure décisive : N reste (G3). A et N+A sont arrêtés à la
graine 0. Limites : 30 % des copies perturbées sont identiques à l'original (une seule phrase,
perturbation sans effet) ; la fiche seule (0,495 au levier 1, autre grille) n'a pas été remesurée
contre N. Le candidat devient N : MiniLM gelé sur le texte normalisé, fiche, régression.

**Levier 3, encodeur réajusté** (`scoring/ml/encodeur.py`, règle écrite le 23/11 avant la mesure,
plan §11). E : MiniLM réajusté de bout en bout sur le PC GPU, même entrée que N (texte normalisé
et fiche), trois têtes, recette fixée d'avance (3 époques, AdamW 2e-5 / 1e-3, lots de 32, 256
jetons). La validation dure est la même (l'apprenant remplace la régression, mêmes plis) ; N est
refait dans la même commande. E doit battre N avec chacune de ses 3 graines, puis tenir ≤ 300 Mo
et ≤ 30 ms par élément en ONNX int8 sur le portable. Dépendances : groupe uv `gpu` (torch CUDA
13.0, transformers, onnx), en conflit avec `nli` (torch processeur) ; ni la CI ni l'image Docker
ne l'installent, et les tests de l'encodeur (petit BERT créé sur place) sont alors sautés.

Sur le PC GPU (pilote NVIDIA compatible CUDA 13, branche `l2/feat/m6-encodeur`) :

```bash
uv sync --all-packages --group gpu
uv run python -c "import torch; print(torch.cuda.is_available())"     # doit afficher True
uv run python -m packages.collect.trainset                             # corpus des gabarits (2 s)
uv run python -m packages.collect.trainset --verifier
uv run python -m packages.collect.trainset.llm_corpus --assembler       # textes du LLM
uv run python -m packages.analyse.scoring.ml.encodeur --sonde           # temps estimé
uv run python -m packages.analyse.scoring.ml.encodeur --levier          # N contre E
uv run python -m packages.analyse.scoring.ml.encodeur --exporter        # ONNX int8 de E0
```

Puis, sur le portable : `uv run python -m packages.analyse.scoring.ml.encodeur --latence <dossier
onnx copié>` (taille, latence par élément, écart int8 − float32 sur le jeu réel).

Mesure du 23/11 (RTX 4060, 27 minutes par graine) :

| F1 macro de l'action | **N, retenue** | E0 | E1 |
|---|---|---|---|
| métiers | **0,851** | 0,945 | 0,946 |
| gabarits tenus à l'écart | **0,716** | 0,772 | 0,731 |
| gabarits → LLM / LLM → gabarits | **0,634 / 0,679** | 0,567 / 0,663 | 0,544 / 0,685 |
| jeu réel | **0,445** | 0,299 | 0,341 |
| précision des alertes au rappel 0,95 | **0,998** | 1,000 | 1,000 |

**Décision de la règle : N reste.** E0 passe la règle (gabarits +0,056 [0,034 ; 0,080]), E1 non
(+0,015 [−0,007 ; 0,038], rien de significatif ailleurs) : arrêt après 2 graines, ni export ni
latence à mesurer. Lecture : le réajustement apprend la formulation du simulateur (+0,09 sur les
métiers, alertes parfaites) et transfère moins bien : gabarits → LLM −0,067 et −0,090, jeu réel
−0,146 [−0,225 ; −0,042] pour E0 et −0,104 [−0,196 ; 0,003] pour E1. Limite de la règle : elle
ne mettait pas de veto sur une perte significative du jeu réel, si bien qu'E0 a « gagné » malgré
elle ; la décision n'en dépend pas ici, mais la règle des prochains leviers doit l'ajouter.

**Mesure finale de N** (règle écrite le 24/11 avant la lecture, plan §11). `--figer-candidat`
apprend N sur tout l'entraînement avec les réglages et le seuil du découpage 1 lus dans le
rapport du levier 2 (`comprendre_candidat.npz`), et refuse d'écrire le modèle s'il ne redonne
pas 0,445 sur le jeu réel. `--mesurer-candidat` lit T-gen et T-démo une seule fois (journal
`journal-geles.jsonl`) et recalcule la référence, déjà mesurée, sur les mêmes cartes pour
l'écart apparié. Lecture du 24/11 (commit 59833f7) :

| F1 macro de l'action | référence | **N** | écart apparié N − référence |
|---|---|---|---|
| T-gen (150 éléments) | 0,625 | **0,769** [0,675 ; 0,845] | **+0,145** [0,072 ; 0,223] |
| T-démo (60 éléments) | 0,164 | **0,270** [0,189 ; 0,364] | **+0,106** [0,008 ; 0,223] |
| alertes de T-gen détectées | 2/2 | 1/2 | alerte −0,152 [−0,503 ; 0,000] |
| alerte de T-démo détectée | 0/1 | 0/1 | 0 |

N est meilleure que la référence sur les deux jeux gelés, de façon significative : la
validation et le test vont dans le même sens. **N devient le modèle de la compréhension pour
M8.** Limites : N manque une des deux alertes de T-gen que la référence trouvait (deux
alertes seulement, écart non significatif, mais l'alerte est la cible qui coûte le plus quand
elle manque : à surveiller dans M8) ; l'alerte de T-démo reste manquée par les deux modèles ;
sur T-démo, 0,270 reste loin du simulateur (décalage de domaine, déjà diagnostiqué). Les
fichiers des mails restent faibles (T-gen : mail 0,528, fichier 0,435).

```bash
uv run python -m packages.analyse.scoring.ml.comprendre --levier-texte  # H, N, A, N+A comparées
uv run python -m packages.analyse.scoring.ml.comprendre --levier-fiche  # R, H, F comparées
uv run python -m packages.analyse.scoring.ml.jeux_m5 --reste            # réponses manquantes
uv run python -m packages.analyse.scoring.ml.comprendre --entrainer     # référence + rapport
uv run python -m packages.analyse.scoring.ml.comprendre --valider-dur   # validation dure
uv run python -m packages.analyse.scoring.ml.annoter --serie m6-dev-reel-echantillon --annotateur theo
uv run python -m packages.analyse.scoring.ml.comprendre --geler-dev     # après l'échantillon
uv run python -m packages.analyse.scoring.ml.comprendre --valider-reel  # jeu de développement réel
uv run python -m packages.analyse.scoring.ml.comprendre --mesurer       # une fois par modèle
```

## Commandes

```bash
uv run pytest packages/analyse --cov=packages/analyse   # tests et couverture
uv run ruff check . && uv run ruff format --check .     # style
```

## Arborescence

```
aggregate/rules.py   règles de sélection (fonctions pures)
aggregate/cases.py   regroupement des items en affaires (L2.1)
brief.py             generate_brief : assemble le Brief
clock.py             heure de Paris et détection des journées entières
guard/check.py       garde déterministe des claims (L2.4)
guard/trace.py       commande trace <brief_id> <rank> (L2.4)
rag/embeddings.py    vecteurs de sens des items (L2.6)
rag/explore.py       outil d'exploration des embeddings sur une journée
rag/index.py         index RAG par utilisateur (L2.6, R1)
rag/retrieve.py      historique des affaires du jour (L2.6, R2)
scoring/score.py     score de priorité par règles (L2.2)
scoring/learned.py   score appris des affaires dans le brief (meilleur mail + tâches, réunions)
scoring/ml/          score appris : données Graph, bruit, intentions, échéances (pipeline A)
scoring/ml/synth/    générateur de mails du lot 2 (SPEC.md)
scoring/ml/annoter.py outil d'annotation unifié (M4)
scoring/ml/jeux_m5.py jeux de mesure gelés (M5)
scoring/ml/comprendre.py référence de M6 et mesure sur les jeux gelés
scoring/annotations/ échantillon et étiquettes de Théo (versionnés, données synthétiques)
tests/               fabrique d'Item synthétiques + tests
```
