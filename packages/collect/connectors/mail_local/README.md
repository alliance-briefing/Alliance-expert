# Partie B : mails locaux

Depuis `C:\Users\thkon\Documents\GitHub\Alliance-expert`, dans l'invite de commande :

```bat
.venv\Scripts\python.exe -m packages.collect.connectors.mail_local --date 2026-10-21
.venv\Scripts\python.exe -m pytest packages/collect/connectors/mail_local
```

Le fichier `sorties/mails.json` contient une liste de fiches `Item` du contrat A.
On peut choisir le fichier avec `--out sorties/autre.json` et le jeu de test avec
`--fixtures chemin/vers/demo`. Aucun compte réel, secret ou API n'est nécessaire.

## Données utilisées

- `contracts/fixtures/demo/manifest.json` : graine, période et managers du jeu livré.
- `collect.synth.generator.build_world` : reconstitution déterministe des mails
  bruts existants, y compris leur historique antérieur aux journées exportées.
- `SimMail` : `native_id`, `sender`, `subject`, `html`, `sent_at`, `answered_at`, etc.
- `collect.synth.window.mail_item` : conversion commune en `Item`, appelant déjà
  `collect.core.text.make_snippet`. Le nettoyeur commun n'est pas modifié.

Les `items.jsonl` livrés sont déjà filtrés et nettoyés ; ils n'exportent ni le HTML
ni `answered_at`. Le programme reconstitue donc les données à partir du manifeste
et du générateur fourni. Un test compare les mails reconstitués aux fiches livrées
pour le 21 octobre. `truth.json` n'est pas utilisé pour décider quels mails garder.

## Règle demandée pour cette version

Référence : heure simulée de la journée, fournie par `collect_time` (08:30:02 Paris).
Les comparaisons de durée se font en UTC pour conserver 24 heures exactes.

1. Mails reçus dans `(référence - 24 heures, référence]`, même déjà répondus.
2. OU mails reçus il y a **au moins 7 jours**, adressés directement au manager,
   provenant d'un humain et toujours sans réponse à la référence (`answered_at`
   absent ou futur). Les anciens robots et mails en copie sont exclus.
3. Aucun mail futur, sortant ou appartenant à un autre manager.
4. Dédoublonnage par `native_id`, tri chronologique, plafond de 500 plus récents.

Cette règle suit la demande actuelle de la partie B. Elle diffère du générateur
commun, qui reporte les mails sans réponse **reçus dans les 7 derniers jours**.
Le code commun reste inchangé. Avec ces deux critères, un mail ne peut pas être
simultanément récent de moins de 24 h et ancien d'au moins 7 jours ; le
dédoublonnage reste utile si les données d'entrée répètent un identifiant.

La date de réception simulée est `sent_at` : aucun champ `received_at` n'existe
dans `SimMail`. `is_read` n'est jamais utilisé pour déduire une réponse.
Les extraits vides et les objets vides sont autorisés par le contrat.
Le nettoyage ne constitue pas une détection des injections de prompt.

Les méthodes de sélection et de conversion sont dans `connector.py`, la commande
dans `__main__.py`, les tests dans `test_mail_local.py`. Le connecteur fonctionne
directement avec cette commande, sans modifier `collect.toml`.
