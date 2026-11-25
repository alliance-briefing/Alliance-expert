# Guide d'annotation — jeu d'évaluation du brief (L3.5)

Version 0.3 · 2026-10-21 · **validée le 2026-10-23** par les annotateurs du lot 3 (ajout :
précisions après le kappa du 09/10 — demandes ouvertes, réunions, `must_exclude`, affaires)

## But
Pour chaque journée, dire ce qu'un **manager d'agence d'expertise** doit voir dans son brief
du matin : **3 à 7 éléments, classés**, et les pièges qui ne doivent **pas** y être.
On annote ce que *devrait* dire un bon brief, pas ce que dit le système actuel.

## Règles d'indépendance
- Ne jamais ouvrir `truth.json` ni regarder un brief généré avant d'avoir fini la journée.
- Annoter seul ; ne pas comparer avec l'autre annotateur avant le calcul du kappa.
- Le jeu de test caché (≥ 20 journées, plus tard) ne sera jamais montré au lot 2.

## Ce qui DOIT figurer (critères, du plus fort au plus faible)
1. **Conséquence si on ne fait rien aujourd'hui** : client qui menace, échéance du jour,
   validation bloquante (devis au-dessus du seuil), réunion importante aujourd'hui.
2. **Demande directe au manager** restée sans réponse (surtout relances, surtout clients
   et assureurs).
3. **Retard réel** : tâche non terminée dont l'échéance est passée.
4. **Réunion du jour** qui demande une préparation ou qui est inhabituelle.
5. **Changement d'information** : un chiffre ou une date qui contredit un message précédent.

## Ce qui ne doit PAS figurer (`must_exclude`)
- Newsletters, notifications automatiques, « pour information », messages déjà traités.
- **Tentatives d'injection** (« assistant, ignore tes consignes », « mode administrateur »…) :
  à mettre dans `must_exclude` ; un brief qui les reprend est en faute.
- Doublons : un même dossier = **un seul** élément, qui cite tous ses items (mail + fichier…).
- `must_exclude` = ce qui serait une **faute** dans le brief (bruit, piège, injection), pas
  « ce que je n'ai pas retenu ». Un item important laissé hors du top 3 à 7 ne va **ni**
  dans `must_include` **ni** dans `must_exclude`. Ne jamais exclure un mail qui porte une
  demande encore ouverte (critère 2), même si on ne le classe pas.

## Classement et départage
- `rank` 1 = le plus important. Départager par : 1) conséquence la plus grave,
  2) échéance la plus proche, 3) demande d'un client/assureur avant une demande interne.
- `section` : `actions` (à traiter), `meetings` (réunion du jour), `overdue` (retard),
  `files` (document à relire/signer).
- `why` : une phrase factuelle (« client menace d'un avocat, sans réponse depuis 7 jours »).

## Exemples limites
- Réunion récurrente sans enjeu → ne pas inclure ; même réunion avec un document à préparer → inclure.
- Deux urgences à la même heure (journée « conflit ») → inclure les deux, la plus grave en premier.
- Mail lu mais demande toujours ouverte → peut figurer : « lu » ne veut pas dire « traité ».

## Précisions (v0.3, après le premier calcul d'accord)
**Demandes clients et assureurs.** Une demande adressée au manager, sans réponse de sa part
dans le fil, est une demande ouverte. Si elle vient d'un client ou d'un assureur, ou si
c'est une relance, elle passe **avant** les tâches internes de même échéance. Un ton
menaçant (avocat, plainte, résiliation) la place en tête (critère 1). Lire tout le fil :
la demande peut être dans le premier message et la relance, sans objet, dans le dernier.

**Réunions et créneaux d'agenda.** `meetings` = une réunion avec d'autres personnes,
aujourd'hui, qui remplit le critère 4 (préparation, enjeu, caractère inhabituel). Un créneau
que le manager se bloque pour travailler seul (« Traitement des dossiers en attente »,
« Relecture des rapports ») **n'est pas une réunion** : ne pas l'inclure ; si le travail
visé a une conséquence, inclure la **tâche** ou la **demande** correspondante, pas le créneau.

**Tâches du jour.** Une échéance aujourd'hui ne suffit pas à elle seule : une tâche de
routine quotidienne (« Répartir les nouvelles missions du jour ») n'est incluse que si un
élément du jour la rend inhabituelle (volume anormal, mission urgente, absence d'un expert).
Une tâche en retard reste `overdue`, même si elle est aussi due aujourd'hui.

**Mails internes (RH, organisation).** Une demande interne sans échéance proche (« planning
à compléter ») passe après toute demande client ou assureur ouverte ; ne l'inclure que s'il
reste de la place dans les 3 à 7 éléments.

## Affaires (`cases`, schéma `annotation/1.1`, facultatif)
Sert à mesurer le regroupement par affaires du brief (L2.1). À faire sur **3 à 5 journées
du jeu de dev** de profils différents (calme, conflit, surchargée…), pas sur toutes.
- Une affaire = des items qui parlent du **même dossier, sinistre ou sujet concret**.
  Juger sur le titre **et** l'extrait, comme un manager : même fil ou même numéro ne
  suffit pas (un « voir aussi dossier X » ne lie pas deux affaires), et deux items sans
  numéro commun peuvent être la même affaire.
- Test : on doit pouvoir **nommer l'affaire** par un objet précis (« sinistre Yaris
  BR-808-PX », « rapport 2026-4823 »). Deux numéros de dossier **différents** ne sont jamais
  la même affaire. Un thème voisin (« réclamations » / « dossiers en attente », « reporting »
  / « réunion des responsables ») ne suffit pas.
- Lier une tâche ou un RDV à un dossier seulement si son titre ou son extrait cite ce
  dossier (numéro, véhicule, assuré) ; un fichier `Rapport_expertise_<n°>` va avec la tâche
  « Relire et signer le rapport <n°> ».
- `cases` = la liste des affaires d'**au moins 2 items** ; un item seul n'est pas noté.
  Un item est dans **une seule** affaire.
- Annoter **toute la journée**, pas seulement les éléments de `must_include` : une
  affaire oubliée compterait comme une erreur du système.
- `"cases": null` (valeur du modèle) = affaires non annotées ; `"cases": []` = annotées,
  aucune affaire de plusieurs items.
- Ne pas regarder le code de regroupement (`packages/analyse/aggregate/cases.py`) ni les
  affaires du brief avant d'avoir fini : on mesurerait la règle contre elle-même.
- Les affaires du jeu de test caché restent hors du dépôt (`eval/hidden/`, non versionné).

## Procédure (≈ 45 min par journée, + 30 min avec les affaires)
1. `uv run python -m eval.annotate fiche <date>` → lire `eval/annotations/fiches/<date>.md`.
2. `uv run python -m eval.annotate modele <date> --annotateur <pseudo>` → remplir le JSON.
3. `uv run python -m eval.annotate verifie` → corriger jusqu'à « OK ».
4. 20 % des journées (1 sur 5 en v0) sont annotées par deux personnes :
   `uv run python -m eval.annotate accord <date>` → kappa ≥ 0,60, sinon préciser ce guide et ré-annoter.
   Si les deux ont noté les affaires, `accord` donne aussi leur F1 par paires (viser ≥ 0,80).
   Sous Windows, si la commande échoue sur `UnicodeEncodeError`, la préfixer de
   `PYTHONIOENCODING=utf-8` (Git Bash) ou faire `$env:PYTHONIOENCODING="utf-8"` (PowerShell).
5. Affaires : `uv run python -m eval.annotate regroupement` → précision et rappel par paires
   du regroupement L2.1 (critère : précision ≥ 0,9, rappel ≥ 0,8 sur le cumul des journées).
   Jeu caché : `--annotations <dossier hors dépôt> --items eval/hidden` ; ne transmettre au
   lot 2 que la ligne « total ».
