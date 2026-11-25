# Limites connues et pistes pour Alliance Experts (L3.9)

Ce qui manque pour une mise en production, et par où reprendre. Premier jet du lot 3
(plateforme), à compléter avec les sections Évaluation et Utilisation (#80).

## Par rapport à la demande

| Livrable du client | État | Écart |
|---|---|---|
| 9 · Moteur LLM de résumé et de hiérarchisation | partiel | Aucun LLM génératif : le brief est composé de titres et de dates (règles), classés par un score lisible ou par un modèle MiniLM local (ADR-0003, ADR-0005). Choix assumé contre l'injection de prompt (un mail piégé ne peut rien écrire dans le brief), mais les « résumés » ne résument pas le contenu des mails. |
| 3 · Connecteurs | partiel | Microsoft seulement ; Google reporté ([ADR-0004](adr/0004-google-reporte.md)). |
| 13 · Authentification et droits | fait, à durcir | Rôles dans `roles.toml`, pas dans les groupes Entra ; la déconnexion ne ferme pas la session Microsoft. |
| 14 · Journalisation | fait, avec trous | Pas d'identifiant de corrélation de bout en bout ni de durée de collecte dans le contrat C ; la lecture du journal par un admin n'est tracée que dans les journaux du serveur. |
| 15 · Évaluation | voir #80 | Résultats et limites décrits par la section Évaluation. |

## Plateforme

- **Un seul processus, sans base de données** ([ADR-0006](adr/0006-stockage-fichiers.md)) :
  sessions, actions en attente et métriques de génération en mémoire ; un redémarrage
  déconnecte tout le monde (sans perte de données). Plusieurs instances demandent Postgres
  (ou équivalent) derrière les interfaces existantes.
- **Pas de HTTPS** : l'application doit être placée derrière un proxy TLS.
- **Latence du score appris** : médiane 2,3 s, 95e centile 7,3 s sur les journées chargées
  (lot 2, M10), au-delà de l'objectif de 3 s ; génération synchrone (pas de tâche de fond).
- **Effacement et conservation** : `delete-user` n'efface ni les briefs ni les retours ; les
  briefs, les retours et le journal de l'API n'ont pas de durée de conservation
  ([securite.md](securite.md)).
- **Mesures** : les durées de collecte ne sont pas mesurées ; un brief qui ne cite aucun élément
  ne se rattache pas à sa génération dans la vue d'administration.
- **Interface d'administration** : fonctionnelle mais sans mise en forme dédiée.

## Pistes, par ordre de valeur pour Alliance Experts

1. **Résumés par un LLM, derrière une interface** : un `BriefGenerator` qui résume le contenu
   des affaires avec un modèle hébergé par Alliance Experts ou local, en gardant la garde des
   claims (chaque phrase citée par sa source) et les tests d'injection du jeu d'évaluation.
2. **Durcir pour la production** : proxy TLS, rôles depuis les groupes Entra, signature de la
   revue de sécurité, durées de conservation et effacement complet d'un utilisateur.
3. **Base de données** pour plusieurs instances et une génération planifiée chaque matin
   (tâche de fond plutôt qu'à la demande), qui ferait aussi disparaître l'attente de 7 s.
4. **Contrat C 1.1** : un identifiant d'exécution et une durée facultatifs, pour suivre une
   collecte jusqu'au brief et mesurer chaque connecteur.
5. **Fournisseur Google**, en suivant le chemin décrit dans l'ADR-0004.
