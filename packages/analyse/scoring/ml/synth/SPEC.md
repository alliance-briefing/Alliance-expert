# Générateur de données du lot 2 — spécification (brouillon du 26/10/2026)

À valider par Théo, Paul (lot 2) et Yacin (référent lot 1). Rien n'est encore écrit.

## Pourquoi
Le score appris (ADR-0003) bute sur le manque de données **variées** : 70 demandes d'action
étiquetées, presque toutes dans deux styles (données de David, démo du lot 1). Deux essais
d'amélioration ont gagné sur ces jeux et reculé sur des mails indépendants (banc CheckList,
transfert vers la démo). Il faut plus de **formulations** différentes, pas plus de mails dans le
même style.

## Principes
1. **Combiner, ne pas réciter.** Un mail = des actes de langage × des formulations × un contexte
   métier × une forme (bruit). Pas de mails entiers recopiés.
2. **Étiquettes par construction.** Le générateur sait quel acte il écrit : « action » et
   « réponse » sont connues sans annotation. Les humains **vérifient un échantillon**.
3. **Plusieurs auteurs.** Les formulations sont écrites séparément par au moins trois personnes
   (Théo, Paul, Claude, et d'autres si possible), chacune dans son fichier, avec son nom.
4. **Ne jamais mesurer sur le générateur.** Il sert à l'entraînement. La mesure se fait sur des
   données écrites par d'autres : David, démo du lot 1, banc CheckList, paires réservées. On ajoute
   une mesure « un auteur laissé de côté » : entraîner sur les formulations de deux auteurs,
   tester sur celles du troisième.
5. **Isolé du lot 1.** Code dans `packages/analyse/scoring/ml/synth/` ; aucun import de
   `packages/collect/synth` ; jamais `truth.json` ni la graine du jeu caché (2027).
6. **Déterministe et fictif** : graine fixée, domaines en `.test`, aucune donnée réelle.

## Niveau 1 — mails isolés (intentions)
### Actes de langage et étiquettes
| Acte | Exemple | action | réponse |
|---|---|---|---|
| Demande explicite | « Peux-tu valider le devis ? », « Merci de signer le bon » | oui | selon la forme |
| Demande implicite | « Il me faudrait le chiffrage jeudi », « Rapport prêt pour relecture » | oui | non |
| Demande à un tiers | « Karim, peux-tu… » (manager en copie), « Je demande à Julie de… » | non | non |
| Engagement de l'expéditeur | « Je reviens vers vous », « Je vous le renvoie demain » | non | non |
| Question | « Le client a-t-il reçu le rapport ? », « Tu es dispo demain ? » | non | oui |
| Proposition de créneau | « Nous sommes disponibles lundi ou mardi, selon ce qui vous arrange » | non | oui |
| Invitation | Invitation à une réunion | non | oui |
| Relance | « Je me permets de vous relancer… » + demande d'origine | comme la demande | comme la demande |
| Information | « Pour info », « Veuillez trouver ci-joint », compte rendu | non | non |
| Confirmation d'un fait | « Le devis a été validé », « Commande bien reçue » | non | non |
| Remerciement, politesse | « Merci, bien reçu », « Bravo pour… » | non | non |
| Alerte (automate) | Connexion inhabituelle, mouvement bancaire | — (alerte = oui) | non |
| Incident | Panne, retard, point bloquant | selon la suite | selon la suite |

Un mail peut combiner plusieurs actes (information puis demande en fin de message) : chaque
étiquette est le « ou » des actes. Une **négation** ou une **annulation** (« pas besoin de
répondre », « finalement inutile ») retire l'étiquette.

### Contextes métier
Cinq ou six secteurs, chacun avec ses objets (dossier, devis, planning, contrat, rapport, ticket…),
ses rôles (N+1, pairs, équipe, clients, fournisseurs, automates) et son vocabulaire : par exemple
cabinet de conseil, réseau d'expertise, BTP, clinique, e-commerce, collectivité.

### Forme et bruit
Salutation variable ou absente ; signatures courtes ou longues (mentions légales) ; citations
« Le … a écrit » ; « RE : » et « TR : » ; fautes et abréviations (« stp », « dsl ») ; environ 10 %
en anglais ; longueur d'une phrase à trois paragraphes ; listes, liens, pièces jointes citées ;
échéances écrites de plusieurs façons, avec la date réelle connue.

### Contrôle
Toi et Paul étiquetez un échantillon de 100 mails générés, sans voir l'étiquette de construction.
Objectif : au moins 90 % d'accord. En dessous, on corrige le générateur, pas les étiquettes.

## Niveau 2 — journées (paires pour le score)
- Une persona par secteur, sur trois semaines ; 5 à 15 affaires qui vivent (demande, relance,
  réponse du manager, clôture) ; **mails envoyés par le manager** (absents chez la démo), pour que
  « déjà répondu » ait un sens ; invitations et réunions ; bruit (lettres d'information, automates).
- Chaque matin à 8 h 30, les mails reçus des trois derniers jours sont les candidats du brief.
- **Les jugements de paires restent humains** : la priorité est subjective, le générateur ne la
  fixe pas. Des paires « évidentes » (alerte de fraude contre lettre d'information) servent de
  contrôle.

## Annotation à deux
- Un fichier d'étiquettes par annotateur (`--annotateur paul`), pour éviter les conflits Git.
- 25 % des items annotés par les deux, sans concertation : accord entre annotateurs (kappa de
  Cohen) ; désaccords tranchés ensemble, qui précisent les critères de la page.
- Le reste est partagé entre les deux.

## Étapes
1. Valider cette spécification (Théo, Paul, Yacin).
2. Écrire les banques de formulations, une par auteur (environ 1 h chacun, 20 à 40 formulations
   par acte).
3. Niveau 1 : générateur de mails isolés, contrôle des 100 mails, entraînement des intentions,
   mesure sur les jeux indépendants et « un auteur laissé de côté ».
4. Niveau 2 : journées, puis une nouvelle série de paires réservées, plus nombreuse, jugée à deux.

## Questions ouvertes
- Yacin : souhaite-t-il que certaines formes du lot 1 (format Graph, pièges) soient reprises,
  pour que le générateur serve aussi à tester les connecteurs ?
- Quels secteurs choisir ? Faut-il en garder un pour le test uniquement (secteur jamais vu) ?
- Paul peut-il aussi écrire des formulations, ou seulement annoter ? (S'il écrit, il ne doit pas
  vérifier ses propres mails.)
