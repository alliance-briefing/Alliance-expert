# ADR-0006 : données de l'application en fichiers dans un volume, sans Postgres pour la v1
Date : 2026-11-25 · Statut : proposé
Décideurs : référents lot 1, 2, 3

## Contexte (3 lignes)
`docker compose` démarrait un service Postgres, mais aucun code ne s'en sert : les Items (lot 1),
le journal de collecte, les retours, le journal de l'API sont déjà des fichiers JSON Lines, rangés
par utilisateur. L3.8 demande des volumes, une sauvegarde et une restauration, et un mode hors ligne.

## Options examinées
1. **Fichiers dans un volume Docker (`/data`).** Rien à écrire de plus pour les données existantes ;
   sauvegarde = une archive du volume ; `delete-user` (L1.7) efface un dossier. Limite : un seul
   processus écrit (un worker), pas de requêtes croisées entre utilisateurs.
2. **Postgres pour tout.** Plusieurs workers et des requêtes SQL, mais 5 magasins à réécrire
   (items, audit ×2, retours, briefs), une migration de schéma, et le lot 1 à modifier.
3. **Garder le service Postgres sans l'utiliser.** Un conteneur de plus à démarrer et à sauvegarder
   pour rien, et un `changeme` comme mot de passe dans le dépôt.

## Décision
Option 1 pour la v1 : service Postgres retiré, un volume `briefing-data` monté sur `/data`.
Chaque magasin reste derrière son interface (`ItemStore`, `AuditSink`, `FeedbackStore`,
`BriefStore`, `SessionStore`) : passer à Postgres ne touche pas les routes.

## Conséquences (ce que ça coûte, comment on revient en arrière)
- Un seul worker uvicorn (déjà le cas) : les sessions et le cache restent dans le processus.
- Sauvegarde et restauration = archive du volume (L3.8, P4).
- Retour arrière : réintroduire le service `db` et une implémentation Postgres de chaque interface,
  magasin par magasin ; à prévoir pour une mise en production à plusieurs instances.
