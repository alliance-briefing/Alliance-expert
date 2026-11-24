# ADR-0004 : fournisseur Google (Gmail, Google Agenda) reporté après la v0.9.0
Date : 2026-11-18 · Statut : accepté (24/11/2026, référent lot 1)
Décideurs : référents lot 1, 2, 3 · à communiquer à David avant le statut du 20/11 (issue #90)

## Contexte (3 lignes)
Le livrable final (L1.10) demande deux fournisseurs pour le mail et le calendrier : Microsoft et
Google. Au 18/11, Microsoft est complet et vérifié sur un vrai compte (mail, calendrier, tâches,
fichiers, OAuth) ; rien n'est commencé côté Google, et le gel des fonctionnalités est le 25/11.

## Options examinées
1. **Gmail + Google Agenda avant le gel.** Le cœur se réutilise (`core/normalize`, `text`,
   `correlation`, `paris_time`, `runner`, `audit`, `retention`) et le contrat A prévoit déjà
   `provider: "google"`. Mais il faut, en 5 jours ouvrés : un projet Google Cloud et un écran de
   consentement (une application non vérifiée est limitée à 100 utilisateurs de test et ses jetons
   de renouvellement expirent au bout de 7 jours), un second flux OAuth (~320 lignes pour
   Microsoft), un client HTTP Gmail (~150 lignes pour `graph_http`), deux connecteurs (~270 et
   ~230 lignes pour leurs équivalents Microsoft) et leurs tests, un second compte de démo à
   remplir. Soit ~1 500 lignes à relire, au moment où les PR Microsoft attendent encore leur
   relecture.
2. **Microsoft seul, sans rien prévoir.** Aucun coût, mais l'écart avec le guide n'est pas
   justifié et rien n'indique comment ajouter Google plus tard.
3. **Reporter Google après la v0.9.0, écart documenté et chemin d'ajout écrit** (recommandé).
   Le client travaille sous Microsoft 365 : c'est le fournisseur de la démo et du rendu. Le
   code est déjà prêt à recevoir une autre source : un connecteur est un dossier de
   `connectors/` découvert automatiquement, et le contrat accepte `provider: "google"`.

## Décision
Option 3. La v0.9.0 et le rendu du 09/12 ne couvrent que Microsoft. Google est un écart assumé
au guide (L1.10), annoncé à David dans le statut du 20/11.

Chemin d'ajout, pour qui reprendra le sujet :
- `core/oauth.py` : ajouter une configuration `[oauth.google]` (code + PKCE, scopes
  `gmail.readonly` et `calendar.readonly`, lecture seule comme pour Microsoft) ;
- `connectors/mail_gmail/` et `connectors/calendar_google/` sur le modèle de `mail_graph` et
  `calendar_graph` : source fichiers pour les tests + API réelle, `provider="google"`, mêmes
  règles de fenêtre, de fuseau, de réessais et d'isolation ;
- `correlation_id` : même règle (titre + heure de début) pour relier invitation et réunion,
  y compris entre fournisseurs.

## Conséquences (ce que ça coûte, comment on revient en arrière)
- Le guide n'est pas suivi sur un point du livrable final : le rendu doit le dire et renvoyer à
  cette fiche. Le lot 2 et le lot 3 ne sont pas touchés (ils ne voient que des Items).
- Le temps gagné va à la relecture et à la robustesse des connecteurs Microsoft (L1.9, #92)
  et à la démo sur le compte fictif.
- Retour arrière : rien à défaire. Si l'équipe ou David veut Google malgré tout, on suit le
  chemin d'ajout ci-dessus (estimation : 3 à 4 jours pour une personne, hors relecture).
