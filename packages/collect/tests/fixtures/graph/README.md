# Réponses Microsoft Graph factices (calendrier)

Copie de `data/calendar/` du jeu **alliance-graph-synth** : chaque fichier reproduit la réponse
de l'appel MCP `ms365` `get-calendar-view` pour une journée (heure de Paris), valeurs en UTC.
Persona fictive : Hélène MARCHAND, Verdier Expertises (domaines `.example`, aucune donnée réelle).
Période : 05/10/2026 → 16/11/2026. Ne pas modifier à la main : régénérer avec le script du jeu.

# Mails (`mail/`, `manifest.json`)

Petit jeu écrit pour les tests du connecteur `mail_graph`, au **même format** que `data/mail/` et
`data/manifest.json` d'alliance-graph-synth (réponse `list-mail-messages`, un fichier par journée de
Paris ; les mails envoyés ont `parentFolderId == sentFolderId`). Pour le jeu complet de David :
`MAIL_GRAPH_DIR=<alliance-graph-synth>/data`.
Chaque message illustre un cas de la fenêtre du 21/10 à 8h30 : reçu dans les 24 h (0001-0003),
direct sans réponse (0004), déjà répondu (0005 + 0105), en copie (0006), plus de 7 jours (0007),
reçu après la collecte (0008), réponse postérieure à la collecte (0009 + 0109), heure d'hiver (0011).
