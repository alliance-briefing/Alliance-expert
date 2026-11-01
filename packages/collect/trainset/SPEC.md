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
du manager. L'écrivain n'expose que ce qu'un vrai connecteur porterait. Un écrivain LLM
(M3) pourra remplacer `writer.py` fait par fait (chaque fait a sa propre graine).

## Règles non négociables

1. **Cloisonné (P1).** Aucun import de `packages.collect.synth` (générateur de l'évaluation,
   jamais lu) ni de `packages.analyse`. Vérifié par `tests/test_trainset_isolation.py`.
2. **Déterministe.** Même graine, mêmes octets : `--verifier` compare au manifeste.
3. **Contrat A strict.** Chaque élément est un `Item` validé. Rien n'est inventé hors du
   contrat : pas d'« assignée par » pour une tâche, pas de lien réunion ↔ fichier. Ces faits
   restent dans le scénario ; le modèle doit les déduire.
4. **Fictif.** Organisations et personnes inventées, domaines en `.test`. Pas d'expertise
   automobile : c'est le domaine de test (T-démo).
5. **L'échéance n'est pas l'urgence.** L'urgence n'est portée que par le ton écrit.

## Contenu

- **16 métiers** (`metiers.py`) : 12 d'entraînement, 4 réservés à T-gen (achats, hôtellerie,
  laboratoire, service client), jamais vus à l'entraînement.
- **40 personas** sur **15 jours ouvrés** (02/11 → 20/11/2026), soit 600 journées :
  3 par métier d'entraînement, 1 par métier réservé. Chacune a un N+1, des pairs, une
  équipe, des clients, des fournisseurs, des automates, et une « diligence » (probabilité de
  traiter une demande dans la journée).
- **10 types d'affaires** : validation, demande client, réunion à préparer, invitation,
  incident, relecture, demande d'information, délégation (manager en copie), tâche
  personnelle, signature. Une affaire traverse les sources (mail, tâche Planner, fichier,
  relance, réponse du manager).
- **Fond** : réunion hebdomadaire, lettres internes, notifications Teams (auteur humain dans
  le texte), tickets, portail RH, fichiers modifiés par le manager, comptes rendus,
  réunions annulées, tâches terminées.
- **Pièges** : injections de prompt (rien + alerte), hameçonnage (faire + urgent + alerte),
  doublons, informations contradictoires, journées calmes (×0,3) et surchargées (×2).
- **Gabarits réservés** : un gabarit sur cinq dans chaque banque n'apparaît qu'en T-gen.

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

## Limites connues (v0.1)

- Pas de mail d'invitation lié à l'événement (`correlation_id` partagé) : seulement l'événement.
- Les gabarits restent répétitifs : c'est le rôle de l'écrivain LLM (M3) d'y remédier.
- La priorité entre éléments n'est pas étiquetée ici : elle vient des jugements par paires.
