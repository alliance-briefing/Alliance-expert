# Corpus d'entraînement simulé (trainset) — spécification

Tâche M1-M2 du plan (`packages/analyse/plan-modele.html`). Écrit à neuf le 01/11/2026 pour
la grille d'actions de M0 (`packages/analyse/scoring/annotations/guide-annotation.md`).

## Principe : les faits d'abord, le texte ensuite

```
Monde (métier, personas, organigramme)        world.py, metiers.py
  └─ Affaires + fond + pièges                 cases.py  (le simulateur)
       └─ Faits : un fait par élément          facts.py
            ├─ Étiqueteur : règles du guide → action, urgence, alerte   facts.label
            └─ Écrivain : fait → Item du contrat A                       writer.py (gabarits)
```

Le simulateur décide des faits, de l'horloge, des liens entre sources et du comportement
du manager. L'écrivain n'expose que ce qu'un vrai connecteur porterait. L'écrivain LLM (M3,
section plus bas) remplace le texte de `writer.py` fait par fait (chaque fait a sa propre
graine), sans toucher aux faits ni aux étiquettes.

## Règles non négociables

1. **Cloisonné (P1).** Aucun import de `packages.collect.synth` (générateur de l'évaluation,
   jamais lu) ni de `packages.analyse`. Vérifié par `tests/test_trainset_isolation.py`.
2. **Déterministe.** Même graine, mêmes octets : `--verifier` compare au manifeste.
3. **Contrat A strict.** Chaque élément est un `Item` validé. Rien n'est inventé hors du
   contrat : pas d'« assignée par » pour une tâche, pas de lien réunion ↔ fichier. Ces faits
   restent dans le scénario ; le modèle doit les déduire.
4. **Fictif.** Organisations et personnes inventées, domaines en `.test`. Pas d'expertise
   automobile : c'est le domaine de test (T-démo).
5. **L'échéance n'est pas l'urgence.** L'urgence est portée par le ton écrit ;
   `native_importance` n'en est qu'un indice bruité (voir « Forme des éléments »).
6. **Forme des connecteurs réels.** Un Item a la forme que lui donnerait le connecteur
   Microsoft du lot 1 : le modèle s'entraîne sur ce qu'il verra en production.

## Contenu

- **16 métiers** (`metiers.py`) : 12 d'entraînement, 4 réservés à T-gen (achats, hôtellerie,
  laboratoire, service client), jamais vus à l'entraînement.
- **40 personas** sur **15 jours ouvrés** (02/11 → 20/11/2026), soit 600 journées :
  3 par métier d'entraînement, 1 par métier réservé. Chacune a un N+1, des pairs, une
  équipe, des clients, des fournisseurs, des automates, et une « diligence » (probabilité de
  traiter une demande dans la journée).
- **10 types d'affaires** : validation, demande client, réunion à préparer, invitation,
  incident, relecture, demande d'information, délégation (manager en copie), tâche
  personnelle, signature. Une affaire traverse les sources (mail, tâche, fichier,
  relance, réponse du manager).
- **Fond** : réunion hebdomadaire, lettres internes, notifications Teams (auteur humain dans
  le texte), tickets, portail RH, fichiers modifiés par le manager, comptes rendus,
  réunions annulées, tâches terminées.
- **Pièges** : injections de prompt (rien + alerte), hameçonnage (faire + urgent + alerte),
  doublons, informations contradictoires, journées calmes (×0,3) et surchargées (×2).
- **Gabarits réservés** : un gabarit sur cinq dans chaque banque n'apparaît qu'en T-gen.

## Forme des éléments (écrivain v0.2.0, alignée le 18/11 sur les connecteurs du lot 1)

| Source | Connecteur imité | Forme |
|---|---|---|
| Mail | `mail_graph` | extrait nettoyé (sans citation ni signature), 500 caractères au plus ; `content_text` toujours vide |
| Réunion | `calendar_graph` | description en extrait ; `content_text` vide |
| Tâche | `tasks_graph` (To Do) | ni participant ni extrait ; échéance à 23h59, heure de Paris ; lien To Do, y compris pour une tâche d'origine Planner |
| Fichier | `files_graph` (OneDrive) | extrait « Dossier/Nom.ext · TYPE · taille · modifié par X », puis « — » et un court contenu pour les docx (gabarits `FILE_TEXT`) |

- **Tâches non collectées.** `tasks_graph` ne lit que les tâches en retard ou dues dans les
  24 h : une tâche sans échéance ou terminée n'est **pas émise**. Son fait reste dans le
  scénario avec `collecte: false` (les liens d'affaire restent lisibles) ; `labels_at` et la
  distribution du manifeste l'ignorent.
- **`native_importance`** (mail, réunion, tâche) : « high » sur 40 % des éléments urgents
  et 3 % des autres, « normal » sinon (tirage propre au fait, qui ne change aucun texte).
- **Hors écrivain** : les fenêtres de collecte (mails des 24 h, réunions des 48 h, fichiers
  des 48 h) relèvent de la vue d'une journée, pas de l'élément ; elles ne sont pas simulées ici.

## Étiquettes

`facts.label` applique le guide au fait :

| Source | Règle |
|---|---|
| Mail | l'acte demandé ; deux actes → le plus engageant ; en copie d'une demande à un autre → S'informer ; envoyé par le manager, doublon, injection → Rien |
| Réunion | annulée → Rien ; il organise ou doit apporter quelque chose → Préparer ; sans réponse, provisoire ou en conflit → Répondre ; il assiste seulement → Rien |
| Tâche | le verbe de l'intitulé ; terminée → Rien |
| Fichier | modifié par lui → Rien ; lié à une tâche → l'action de la tâche ; lié à une réunion → Préparer ; à signer → Décider ; à relire → Relire ; sinon S'informer |

L'étiquette dépend du jour : `facts.label_on(ligne, moment)` rend Rien une fois l'élément
traité (`resolved_at`) ou la réunion passée (`ends_at`), et None avant `visible_from`.

## Sorties et commandes

```bash
uv run python -m packages.collect.trainset              # génère data/ (≈ 2 s)
uv run python -m packages.collect.trainset --verifier   # compare au manifeste versionné
```

- `data/items/<persona>.jsonl` : les `Item` (non versionné, régénérable) ;
- `data/scenario/<persona>.jsonl` : faits et étiquettes (non versionné) ;
- `data/manifest.json` : empreintes SHA-256 et distribution (versionné) ;
- `data/sample/` : 2 jours de `logistique-1` et `hotel-1` (versionné, pour lire sans générer).

Le lot 2 lit le corpus par `load.py` (`personas`, `load_items`, `load_scenario`,
`labels_at`) et ne génère rien lui-même.

## Écrivain LLM (M3) : le texte des mails, pas les faits

Le LLM **écrit**, il ne **simule** pas : faits, horloge, liens et étiquettes viennent toujours
du simulateur. Hors produit : il ne sert qu'à produire le corpus d'entraînement, sur un PC à
carte graphique (Ollama, Gemma 2 9B), en local ; le brief n'utilise aucun LLM génératif.

**Ce qui est réécrit** (`llm_writer.rewritable`) : les mails rédigés par gabarits, soit 8 152
mails des 40 personas. Automates, pièges, messages fixes et réponses du manager gardent leurs
gabarits : leur forme fait partie du fait.

**Consigne** (`llm_writer.py`, versionnée : `PROMPT_VERSION`, aujourd'hui `mail-v8`) :
- une **fiche** en mots simples, tirée par fait : expéditeur et relation, destinataire (ou
  copie), tu ou vous, sujet, contexte de l'affaire, demande avec la personne qui agit, forme,
  échéance, ton, style ; jamais le nom de l'étiquette ;
- deux **exemples écrits à la main** (`llm_exemples.py`), d'autres actes que le fait ; un mail
  sans demande voit d'abord l'exemple sans demande (v8) ;
- le modèle rend des **morceaux** (objet, contexte, demande, ton, conclusion) en JSON ; le code
  assemble salutation, ordre, second acte et signature (v7) ;
- température 0,8, graine par fait et par essai, 3 essais au plus, réécriture guidée par les
  raisons du refus.

**Contrôles** (règles, jamais un LLM) : par morceau puis sur le mail entier, dans l'extrait de
500 caractères que verra le modèle : acte demandé présent, aucun acte plus engageant ajouté,
échéance et ton écrits, aucun ton non demandé, forme respectée, rôle non inversé, tu ou vous
tenu, rien d'inventé (noms, organisations, chiffres, dates, liens), français, pas d'intitulé de
la fiche recopié. Un morceau facultatif qui échoue est retiré ; un mail « Rien » perd toujours
contexte et conclusion (ils y ajoutaient un fait). Un texte refusé garde le texte des gabarits.

**Production et assemblage** (`llm_corpus.py`) :
- génération sur plusieurs nuits, dans un ordre mélangé à graine fixe (une partie écrite est
  un échantillon au hasard du tout), reprise après arrêt → `data/llm/<consigne>/textes.jsonl`
  (une ligne par fait : morceaux, mail, extrait, essais, raisons, modèle et empreinte) ;
- assemblage **sans GPU** (`--assembler`, client hors ligne qui refuse tout appel) : rejoue le
  simulateur et écrit `items/`, `scenario/`, `manifest.json` dans la forme de `data/`, lisible
  par `load.py` avec `data=<dossier>` ;
- **recontrôle** : le mail enregistré doit s'expliquer par la fiche et l'assemblage d'ici
  (preuve que les deux PC concordent) ; les contrôles d'aujourd'hui sont rejoués sur les
  morceaux : un morceau requis refusé fait retomber sur les gabarits, un facultatif retiré
  raccourcit le mail. Statuts : `retenu`, `rejete`, `recontrole`, `absent` (marqués dans le
  scénario) ;
- versionné : `textes.jsonl.gz` (`--compresser`, stable octet pour octet) et le manifeste ;
  `items/` et `scenario/` se régénèrent.

**Acceptation** (plan, M3), outils côté lot 2 (`packages/analyse/scoring/ml/sonde_m3.py`) :

Corpus final (`mail-v8`, 21/11/2026) : 8 152 mails rédigés, 7 655 textes du LLM retenus,
422 rejetés à la rédaction et 75 au recontrôle (texte des gabarits), 1 047 mails « Rien »
raccourcis. Les échantillons humains ont été tirés sur les 4 114 premiers textes, identiques
octet pour octet aux 4 114 premières lignes du fichier final.

| Critère | Mesure | Résultat |
|---|---|---|
| ≥ 8 000 éléments rédigés | manifeste de l'assemblage | 8 152 rédigés, 7 655 retenus |
| ≥ 95 % passent les contrôles | manifeste de l'assemblage | **93,9 %** (train 93,9 %, T-gen 93,6 %) : non atteint, après durcissement des contrôles d'après l'échantillon (95,0 % avant) |
| bien plus de modèles de phrases qu'avec les gabarits (ADR-0003) | `--diversite`, mêmes 7 655 mails, objets, noms, nombres, dates masqués | **×9,9** (14 267 contre 1 444) ; 79 % de phrases uniques contre 17 % ; objets : 2 640 modèles contre 19 |
| ≥ 90 % conformes sur 5 % relus à l'aveugle | `--echantillon`, `--conformite` | 83,7 % brut (387 cartes, 2 annotateurs, kappa action 0,92) ; 95,8 % après correction et relecture (optimiste) |
| confirmation sur des textes jamais vus | `--confirmation` (150 cartes, critère écrit avant) | **93,3 %** (IC 95 % 88,2-96,3 %), atteint |

Historique des consignes : v2 sonde 14/20 → v7 16/20 (sonde) et 87,7 % de contrôles sur 919
textes → v8 92,7 % sur les mêmes 150 faits (81 % du premier coup, 8 s par texte).

Commandes (PC à carte graphique, puis portable) :

```bash
uv run python -m packages.collect.trainset.llm_corpus                   # générer (reprend)
uv run python -m packages.collect.trainset.llm_corpus --comparer mail-v7 mail-v8
uv run python -m packages.collect.trainset.llm_corpus --assembler       # sans GPU
uv run python -m packages.collect.trainset.llm_corpus --compresser
```

## Limites connues (v0.2, M3)

- Pas de mail d'invitation lié à l'événement (`correlation_id` partagé) : seulement l'événement.
- La priorité entre éléments n'est pas étiquetée ici : elle vient des jugements par paires.
- Écrivain LLM : seuls les mails sont réécrits ; réunions, tâches et fichiers gardent leurs
  gabarits.
- Fiche du simulateur : « que X dise si {o} lui convient » ou « donne son avis » sur une
  commande, un devis ou un budget se lit Décider (9 écarts sur 537 textes relus) ; les
  formulations « appelle » et « choisisse entre » passent moins souvent les contrôles.
- Mails en copie : la règle du guide « copie → S'informer » (19/11) n'est pas encore validée ;
  l'annotateur lit Rien quand le texte ne dit pas que le manager est en copie pour suivre.
- Mesures humaines de M3 à un seul annotateur pour la relecture et la confirmation.
