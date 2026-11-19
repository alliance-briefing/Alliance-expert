"""Exemples écrits à la main pour l'écrivain LLM : une fiche, le mail, et ses morceaux.

Deux exemples sont montrés au modèle avant chaque fiche (jamais du même acte que le fait à
écrire) : ils lui montrent ce qu'est un mail réaliste (un contexte, une demande reformulée,
une conclusion), bien mieux qu'une règle. Les noms et les organisations viennent du monde
simulé : un texte qui les recopierait hors de sa fiche est rejeté par les contrôles (nom ou
organisation inventés), comme un texte qui reprend 8 mots de suite d'un exemple.

Depuis mail-v7, le modèle n'écrit plus le mail entier mais ses morceaux (`MORCEAUX` : objet,
contexte, demande, ton, conclusion) ; le code les assemble. Les fiches suivent le format de
`llm_writer.sheet` (la ligne « Style » des fiches ci-dessous est raccourcie à la volée).
"""

from __future__ import annotations

EXAMPLES: tuple[dict, ...] = (
    {
        "id": "decider-indirecte",
        "act": "decider",
        "fiche": """Entreprise : TransAlpes
De : Romain Benali, Responsable d'entrepôt — pour Sarah : un membre de son équipe
À : Sarah Lemoine, Responsable logistique (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : le contrat de transport frigorifique
Contexte : Romain a préparé le contrat de transport frigorifique ; sans décision, la suite est en attente.
Demande : que Sarah signe le contrat de transport frigorifique.
Qui agit : c'est Sarah qui décide ; Romain attend sa décision et ne décide pas lui-même.
Forme : indirecte (ne formule pas la demande : décris ce qui manque ou ce qui bloque)
Échéance : jeudi 12 novembre
Ton : neutre (aucune formule de délai, ni urgente ni calme)
À écrire : « jeudi 12 novembre »
Style : direct et efficace ; commence par une nouvelle du dossier ; 70 à 120 mots
Signature : Romain""",
        "objet": "Contrat frigorifique : tout est prêt de notre côté",
        "corps": """Bonjour Sarah,

Bonne nouvelle sur le contrat de transport frigorifique : le transporteur a intégré nos remarques sur les créneaux de livraison, et la version reçue hier soir est propre. Je l'ai reprise avec l'équipe de quai, rien ne coince côté exploitation.

Il ne manque plus que ta signature. Tant qu'elle n'y est pas, je ne peux pas confirmer les tournées au transporteur, qui attend notre retour pour le jeudi 12 novembre.

Le contrat est rangé dans le dossier partagé, avec les remarques intégrées.

Romain""",
    },
    {
        "id": "faire-question-vous",
        "act": "faire",
        "fiche": """Entreprise : Valmont Agroalimentaire
De : Nadia Fontaine, Client, Super Marché Ouest — pour Hugo : un client externe
À : Hugo Masson, Directeur des opérations (destinataire direct)
Adresse : vouvoiement (vous, votre, vos) du début à la fin, jamais tu
Sujet : le rapport d'audit hygiène
Contexte : le client a besoin de cet élément pour avancer de son côté.
Demande : que Hugo envoie le rapport d'audit hygiène à Nadia.
Qui agit : c'est Hugo qui doit agir ; Nadia ne le fait pas elle-même et ne propose pas de le faire.
Forme : question
Échéance : aucune, n'en mets pas
Ton : neutre (aucune formule de délai, ni urgente ni calme)
Style : formel et précis ; commence par situer le dossier ; 70 à 120 mots
Signature : Nadia Fontaine, Super Marché Ouest""",
        "objet": "Dossier fournisseur – rapport d'audit hygiène",
        "corps": """Bonjour Monsieur Masson,

Nous préparons le référencement de vos produits pour la saison prochaine, et notre service qualité doit réunir un dossier fournisseur complet avant de le présenter en comité interne.

Il nous manque encore le rapport d'audit hygiène de votre site. Pourriez-vous nous le faire parvenir ? Une version numérique suffira, notre équipe se chargera de l'intégrer au dossier.

Je vous remercie par avance et vous souhaite une bonne journée.

Nadia Fontaine
Super Marché Ouest""",
    },
    {
        "id": "repondre-question-vous",
        "act": "repondre",
        "fiche": """Entreprise : Valmont Agroalimentaire
De : Sylvain Mercier, Fournisseur, Emballages Durand — pour Céline : un fournisseur externe
À : Céline Barbier, Directrice des opérations (destinataire direct)
Adresse : vouvoiement (vous, votre, vos) du début à la fin, jamais tu
Sujet : le cahier des charges des barquettes
Contexte : Sylvain fait le point sur plusieurs dossiers en cours.
Demande : que Céline dise si le cahier des charges des barquettes lui convient.
Qui agit : c'est Céline qui répond ; Sylvain attend sa réponse.
Forme : question
Échéance : vendredi 13 novembre
Ton : neutre (aucune formule de délai, ni urgente ni calme)
À écrire : « vendredi 13 novembre »
Style : cordial et posé ; commence par rappeler votre dernier échange, sans date ; 70 à 110 mots
Signature : Sylvain Mercier, Emballages Durand""",
        "objet": "Cahier des charges des barquettes – nouvelle version",
        "corps": """Bonjour Madame Barbier,

Lors de notre dernier échange, vous m'aviez fait part de vos remarques sur l'épaisseur des barquettes et sur leur tenue au froid. Nous avons repris le cahier des charges des barquettes en conséquence, et notre atelier a confirmé la faisabilité des modifications.

Avant de lancer la fabrication des échantillons, pourriez-vous me dire si cette nouvelle version vous convient ? Une réponse d'ici le vendredi 13 novembre nous permettrait de tenir le calendrier prévu.

Bien cordialement,

Sylvain Mercier
Emballages Durand""",
    },
    {
        "id": "repondre-relance-urgent",
        "act": "repondre",
        "fiche": """Entreprise : Numéria Services
De : Julie Perrin, Responsable infrastructure — pour Thomas : un pair (même niveau, autre service)
À : Thomas Robin, Chef de projet informatique (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : la bascule du serveur de fichiers
Contexte : c'est une relance : Julie a déjà écrit il y a deux jours et n'a pas eu de réponse.
Demande : que Thomas propose une date pour la bascule du serveur de fichiers.
Qui agit : c'est Thomas qui répond ; Julie attend sa réponse.
Forme : directe (impératif ou « merci de… »), sans question
Échéance : aucune, n'en mets pas
Ton : urgent
À écrire : « ça bloque »
Style : pressé, phrases courtes ; rappelle d'abord le message resté sans réponse ; 45 à 70 mots
Signature : Julie""",
        "objet": "Relance – bascule du serveur de fichiers",
        "corps": """Salut Thomas,

Je reviens vers toi pour la bascule du serveur de fichiers : mon message d'avant-hier est resté sans réponse. De notre côté, l'équipe est prête et le matériel est en place.

Là, ça bloque : sans date de ta part, je ne peux pas réserver l'intervention. Donne-moi une date pour la bascule, même approximative.

Julie""",
    },
    {
        "id": "relire-calme",
        "act": "relire",
        "fiche": """Entreprise : Solvane Industrie
De : Camille Roussel, Chargée de recrutement — pour Nicolas : un membre de son équipe
À : Nicolas Henry, DRH (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : la procédure d'accueil des nouveaux arrivants
Contexte : Camille a terminé une version presque finale de la procédure d'accueil des nouveaux arrivants.
Demande : que Nicolas relise la procédure d'accueil des nouveaux arrivants.
Qui agit : c'est Nicolas qui doit agir ; Camille ne le fait pas elle-même et ne propose pas de le faire.
Forme : directe (impératif ou « merci de… »), sans question
Échéance : aucune, n'en mets pas
Ton : calme, sans urgence
À écrire : « rien ne presse »
Style : chaleureux et un peu bavard ; commence par une nouvelle du dossier ; 120 à 180 mots
Signature : Camille""",
        "objet": "Procédure d'accueil : la nouvelle version est prête",
        "corps": """Bonjour Nicolas,

J'ai enfin terminé la nouvelle procédure d'accueil des nouveaux arrivants ! Avant de la diffuser aux équipes, j'aimerais beaucoup que tu la relises. Rien ne presse, la prochaine arrivée n'est pas pour tout de suite.

Après les retours des managers d'atelier, j'ai revu toute la partie sur la première semaine : la visite du site, la remise des équipements et le rendez-vous avec le tuteur sont maintenant décrits étape par étape, avec une liste à cocher. J'ai aussi simplifié le passage sur les accès informatiques, qui était devenu illisible au fil des versions.

Tu as une vision plus large que moi des sujets sensibles, et ton regard sur le ton général me sera précieux. Tu peux annoter directement dans le document partagé.

Merci d'avance,
Camille""",
    },
    {
        "id": "preparer-reunion",
        "act": "preparer",
        "fiche": """Entreprise : Bâtir Ensemble
De : Olivier Dumont, Conducteur de travaux principal — pour Inès : son N+1 (supérieur hiérarchique)
À : Inès Faure, Cheffe de chantier (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : le planning des finitions
Contexte : la réunion du mardi 17 novembre réunit l'équipe et la direction.
Demande : que Inès présente le planning des finitions à la réunion du mardi 17 novembre (dix minutes).
Qui agit : c'est Inès qui doit agir ; Olivier ne le fait pas lui-même et ne propose pas de le faire.
Forme : directe (impératif ou « merci de… »), sans question
Échéance : aucune, n'en mets pas
Ton : neutre (aucune formule de délai, ni urgente ni calme)
Style : cordial et posé ; entre directement dans le sujet après la salutation ; 70 à 120 mots
Signature : Olivier""",
        "objet": "Réunion de chantier – ton point sur les finitions",
        "corps": """Bonjour Inès,

Pour la réunion du mardi 17 novembre, j'aimerais que tu présentes le planning des finitions. Prévois une dizaine de minutes : le maître d'ouvrage sera là et voudra comprendre l'enchaînement des corps de métier sur les derniers étages.

L'idée n'est pas de tout détailler, mais de montrer les grandes étapes et les points de vigilance que tu vois venir, notamment la coordination entre peintres et électriciens.

La salle sera équipée d'un écran pour tes supports.

Bonne journée,
Olivier""",
    },
    {
        "id": "informer-point",
        "act": "informer",
        "fiche": """Entreprise : Clinique des Cèdres
De : Aurélie Bonnet, Responsable pharmacie — pour Marc : un pair (même niveau, autre service)
À : Marc Leroy, Cadre de santé (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : la commande de matériel de perfusion
Contexte : Aurélie fait un point d'étape.
Information donnée : la commande de matériel de perfusion est partie chez le fournisseur. Marc n'a rien à faire ni à répondre.
Ton : neutre (aucune formule de délai, ni urgente ni calme)
Style : direct et efficace ; commence par situer le dossier ; 45 à 70 mots
Signature : Aurélie""",
        "objet": "Matériel de perfusion : commande passée",
        "corps": """Bonjour Marc,

Suite à notre échange sur les ruptures de la semaine dernière, je te confirme que la commande de matériel de perfusion est partie chez le fournisseur ce matin. Les quantités tiennent compte de la consommation de ton service.

Tu n'as rien à faire de ton côté, je m'occupe du suivi de la livraison.

Aurélie""",
    },
    {
        "id": "rien-merci",
        "act": "rien",
        "fiche": """Entreprise : Precisium
De : Lucas Girard, Technicien qualité — pour Émilie : un membre de son équipe
À : Émilie Brunet, Responsable qualité (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : le rapport de non-conformité
Rien n'est demandé : Lucas remercie ou accuse réception concernant le rapport de non-conformité, sans nouvelle information.
Ton : neutre (aucune formule de délai, ni urgente ni calme)
Style : chaleureux ; 35 à 60 mots
Signature : Lucas""",
        "objet": "Re: rapport de non-conformité",
        "corps": """Salut Émilie,

Merci pour ton retour sur le rapport de non-conformité, c'est bien noté. Tes remarques sur l'analyse des causes étaient très claires, je les garde en tête pour les prochains rapports.

Bonne fin de journée,
Lucas""",
    },
    {
        "id": "faire-urgent-tiers",
        "act": "faire",
        "fiche": """Entreprise : Groupe Ardent
De : Pierre Colin, Président — pour Anaïs : son N+1 (supérieur hiérarchique)
À : Anaïs Vidal, Directrice administrative et financière (destinataire direct)
Adresse : tutoiement (tu, te, ton, ta, tes) du début à la fin, jamais vous
Sujet : l'attestation de trésorerie
Contexte : Pierre doit répondre à son tour à la banque et a besoin de cet élément.
Demande : que Anaïs transmette l'attestation de trésorerie à Yanis Caron.
Qui agit : c'est Anaïs qui doit agir ; Pierre ne le fait pas lui-même et ne propose pas de le faire.
Forme : question
Échéance : aucune, n'en mets pas
Ton : urgent
À écrire : « au plus vite »
Style : pressé, phrases courtes ; entre directement dans le sujet après la salutation ; 45 à 70 mots
Signature : Pierre""",
        "objet": "Attestation de trésorerie pour la banque",
        "corps": """Anaïs,

Yanis Caron m'a relancé ce matin : la banque attend toujours l'attestation de trésorerie pour finaliser son dossier. Peux-tu la lui transmettre au plus vite ? Il est joignable toute la journée.

Je compte sur toi, c'est le dernier document qui manque.

Pierre""",
    },
)

# Les mêmes exemples, en morceaux (mail-v7) : ce que le modèle doit rendre pour chaque fiche.
MORCEAUX: dict[str, dict[str, str]] = {
    "decider-indirecte": {
        "objet": "Contrat frigorifique : tout est prêt de notre côté",
        "contexte": "Le transporteur a intégré nos remarques sur les créneaux de livraison, et la version reçue hier soir est propre. Je l'ai reprise avec l'équipe de quai, rien ne coince côté exploitation.",
        "demande": "Il ne manque plus que ta signature sur le contrat de transport frigorifique : sans elle, je ne peux pas confirmer les tournées, et le transporteur attend notre retour pour le jeudi 12 novembre.",
        "ton": "",
        "conclusion": "Le contrat est rangé dans le dossier partagé, avec les remarques intégrées.",
    },
    "faire-question-vous": {
        "objet": "Dossier fournisseur – rapport d'audit hygiène",
        "contexte": "Nous préparons le référencement de vos produits pour la saison prochaine, et notre service qualité doit réunir un dossier fournisseur complet avant le comité interne.",
        "demande": "Pourriez-vous nous faire parvenir le rapport d'audit hygiène de votre site ?",
        "ton": "",
        "conclusion": "Une version numérique suffira, notre équipe se chargera de l'intégrer au dossier.",
    },
    "repondre-question-vous": {
        "objet": "Cahier des charges des barquettes – nouvelle version",
        "contexte": "Suite à vos remarques sur l'épaisseur des barquettes et leur tenue au froid, nous avons repris le cahier des charges, et notre atelier a confirmé la faisabilité des modifications.",
        "demande": "Pourriez-vous me dire si cette nouvelle version du cahier des charges des barquettes vous convient, d'ici le vendredi 13 novembre ?",
        "ton": "",
        "conclusion": "Nous pourrons ensuite lancer la fabrication des échantillons.",
    },
    "repondre-relance-urgent": {
        "objet": "Relance – bascule du serveur de fichiers",
        "contexte": "Mon message d'avant-hier sur la bascule du serveur de fichiers est resté sans réponse. De notre côté, l'équipe est prête et le matériel est en place.",
        "demande": "Donne-moi une date pour la bascule, même approximative, que je puisse réserver l'intervention.",
        "ton": "Là, ça bloque : sans cette date, tout est à l'arrêt.",
        "conclusion": "",
    },
    "relire-calme": {
        "objet": "Procédure d'accueil : la nouvelle version est prête",
        "contexte": "J'ai terminé la nouvelle procédure d'accueil des nouveaux arrivants : la première semaine est maintenant décrite étape par étape, avec une liste à cocher.",
        "demande": "Avant de la diffuser aux équipes, j'aimerais que tu relises la procédure d'accueil des nouveaux arrivants, en annotant directement le document partagé.",
        "ton": "Rien ne presse, la prochaine arrivée n'est pas pour tout de suite.",
        "conclusion": "Ton regard sur le ton général me sera précieux.",
    },
    "preparer-reunion": {
        "objet": "Réunion de chantier – ton point sur les finitions",
        "contexte": "Le maître d'ouvrage sera présent à la prochaine réunion et voudra comprendre l'enchaînement des corps de métier sur les derniers étages.",
        "demande": "Présente le planning des finitions à la réunion du mardi 17 novembre : prévois une dizaine de minutes sur les grandes étapes et les points de vigilance.",
        "ton": "",
        "conclusion": "La salle sera équipée d'un écran pour tes supports.",
    },
    "informer-point": {
        "objet": "Matériel de perfusion : commande passée",
        "contexte": "Suite à notre échange sur les ruptures de la semaine dernière, j'ai revu les quantités avec la consommation de ton service.",
        "demande": "La commande de matériel de perfusion est partie chez le fournisseur ce matin : tu n'as rien à faire de ton côté.",
        "ton": "",
        "conclusion": "Je m'occupe du suivi de la livraison.",
    },
    "rien-merci": {
        "objet": "Re: rapport de non-conformité",
        "contexte": "",
        "demande": "Merci pour ton retour sur le rapport de non-conformité, c'est bien noté.",
        "ton": "",
        "conclusion": "Tes remarques sur l'analyse des causes étaient très claires.",
    },
    "faire-urgent-tiers": {
        "objet": "Attestation de trésorerie pour la banque",
        "contexte": "Yanis Caron m'a relancé ce matin : la banque attend toujours l'attestation de trésorerie pour finaliser son dossier.",
        "demande": "Peux-tu lui transmettre l'attestation de trésorerie ?",
        "ton": "C'est à faire au plus vite, c'est le dernier document qui manque.",
        "conclusion": "Il est joignable toute la journée.",
    },
}  # fmt: skip
