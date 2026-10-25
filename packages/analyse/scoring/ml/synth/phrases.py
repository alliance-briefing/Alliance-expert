"""Banques de formulations par acte de langage (auteur : Claude, 26/10/2026).

Champs remplis par le générateur : {v} verbe à l'infinitif, {o} objet (« le devis du lot
plomberie »), {O} objet avec majuscule, {quand} échéance, {quand_moi} délai de l'expéditeur,
{tiers} prénom d'un tiers, {jour} et {jour2} jours, {heure}, {reunion}, {projet}.
« [tu|vous] » : la variante est choisie selon le registre (tutoiement ou vouvoiement).

Étiquettes par construction : `ACTS[acte] = (action, reponse)` ; None = à juger (une demande
posée en question attend-elle une réponse ? cela dépend du ton).
"""

from __future__ import annotations

ACTS: dict[str, tuple[bool | None, bool | None]] = {
    "demande explicite": (True, None),
    "demande implicite": (True, False),
    "relance": (True, None),
    "demande à un tiers": (False, False),
    "engagement de l'expéditeur": (False, False),
    "question": (False, True),
    "proposition de créneau": (False, True),
    "invitation": (False, True),
    "information": (False, False),
    "confirmation": (False, False),
    "remerciement": (False, False),
}
# Modificateurs : ils annulent une étiquette posée par un autre acte du même mail.
CANCELS = {"annulation": "action", "sans réponse": "reponse"}

BANK: dict[str, tuple[str, ...]] = {
    "demande explicite": (
        "Merci de {v} {o} {quand}.",
        "[Peux-tu|Pouvez-vous] {v} {o} {quand} ?",
        "[Pourrais-tu|Pourriez-vous] {v} {o} ?",
        "[Tu peux|Vous pouvez] {v} {o} {quand} ?",
        "Je [te|vous] demande de {v} {o} {quand}.",
        "[Merci de bien vouloir|Je vous remercie de bien vouloir] {v} {o}.",
        "[N'oublie pas|N'oubliez pas] de {v} {o} {quand}.",
        "[Pense|Pensez] à {v} {o} {quand}.",
        "[Je compte sur toi pour|Je compte sur vous pour] {v} {o} {quand}.",
        "Est-ce que [tu pourrais|vous pourriez] {v} {o} {quand} ?",
        "[Ce serait sympa de|Je vous serais reconnaissant de] {v} {o} {quand}.",
        "À faire de [ton|votre] côté : {v} {o} {quand}.",
        "Une petite demande : [peux-tu|pourriez-vous] {v} {o} ?",
        "Je [te|vous] laisse {v} {o} et me dire quand c'est fait.",
        "Action attendue de [ta|votre] part : {v} {o} {quand}.",
        "[Peux-tu|Pouvez-vous] [t'occuper|vous occuper] de {o} {quand} ?",
        "J'ai besoin que [tu t'en occupes|vous vous en occupiez] {quand} : {v} {o}.",
        "[Il faudrait que tu puisses|Il faudrait que vous puissiez] {v} {o} {quand}.",
        "[Tu me fais signe|Vous me faites signe] quand {o} est prêt ? Il faut le {v} {quand}.",
        "Urgent : merci de {v} {o} aujourd'hui.",
    ),
    "demande implicite": (
        "Il me faudrait [ton|votre] accord sur {o} {quand}.",
        "J'attends encore [ton|votre] feu vert sur {o} pour avancer.",
        "Sans [ta|votre] validation {quand}, nous ne pourrons pas lancer {projet}.",
        "Il manque [ta|votre] signature sur {o}.",
        "{O} est en attente de [ta|votre] validation depuis lundi.",
        "Ce serait bien d'avoir [ton|votre] retour sur {o} {quand}.",
        "Tout est prêt de notre côté, il ne manque plus que [ton|votre] accord sur {o}.",
        "Le document est déposé dans le dossier partagé, prêt pour [ta|votre] relecture : {o}.",
        "{O} attend [ton|votre] arbitrage.",
        "Pour boucler {projet}, il nous faut [ta|votre] décision sur {o} {quand}.",
        "Je n'ai toujours pas [ton|votre] retour sur {o}, et l'échéance approche.",
        "Nous sommes bloqués tant que {o} n'a pas [ton|votre] visa.",
        "{O} est à valider de [ton|votre] côté {quand}.",
        "Le client attend notre position sur {o} {quand} et [c'est toi qui as|c'est vous qui avez] la main.",
        "Dernière étape avant l'envoi : relire {o}, de [ton|votre] côté.",
        "La version finale de {o} n'attend plus que [ta|votre] signature.",
        "Il reste [ton|votre] visa à apposer sur {o} avant {reunion}.",
        "Le prestataire ne peut pas démarrer sans [ton|votre] bon pour accord sur {o}.",
    ),
    "relance": (
        "Je me permets de [te|vous] relancer : ",
        "Petite relance : ",
        "Sauf erreur de ma part, je n'ai pas eu de retour. ",
        "Je reviens vers [toi|vous] concernant ma demande de la semaine dernière : ",
        "Deuxième relance, c'est assez urgent : ",
        "Je [te|vous] remonte ce message : ",
    ),
    "demande à un tiers": (
        "[Je te mets en copie|Je vous mets en copie] : {tiers}, peux-tu {v} {o} {quand} ?",
        "[Je te mets en copie|Je vous mets en copie] : {tiers}, merci de {v} {o} {quand}.",
        "Je demande à {tiers} de {v} {o}.",
        "Je transmets à {tiers}, qui va {v} {o} {quand}.",
        "{tiers} se charge de {v} {o}.",
        "C'est {tiers} qui va {v} {o}, je [te|vous] tiens au courant.",
        "J'ai demandé à {tiers} de {v} {o} {quand}.",
        "{tiers} va {v} {o} dans la journée.",
        "{tiers}, tu peux t'en occuper ? [Je te mets en copie pour info|Je vous mets en copie pour information], rien à faire de [ton|votre] côté.",
        "Je laisse {tiers} {v} {o}, [tu n'as rien à faire|vous n'avez rien à faire].",
    ),
    "engagement de l'expéditeur": (
        "Je regarde ça et je [reviens vers toi|reviens vers vous] {quand_moi}.",
        "Je [te|vous] renvoie {o} {quand_moi}.",
        "Je m'occupe de {v} {o} {quand_moi}.",
        "Je [te|vous] tiens au courant dès que j'ai du nouveau.",
        "Nous [te|vous] transmettons {o} {quand_moi}.",
        "Je vais {v} {o} et je [te|vous] confirme.",
        "C'est noté, je fais le nécessaire.",
        "Je prends en charge {o}.",
        "Je [t'envoie|vous envoie] la version corrigée {quand_moi}.",
        "On s'en occupe, pas d'inquiétude.",
        "Je [te|vous] fais un retour {quand_moi}.",
        "Je finalise {o} et je [te|vous] l'envoie {quand_moi}.",
        "Bien noté, je m'en charge.",
        "Je [te|vous] prépare une proposition {quand_moi}.",
    ),
    "question": (
        "Le client a-t-il bien reçu {o} ?",
        "Où en est-on sur {projet} ?",
        "[As-tu|Avez-vous] eu le temps de regarder {o} ?",
        "[Tu as|Vous avez] des nouvelles concernant {o} ?",
        "Qui est le bon interlocuteur pour {o} ?",
        "[Tu es dispo|Vous êtes disponible] {jour} à {heure} pour faire le point ?",
        "Quel est [ton|votre] avis sur {o} ?",
        "[Tu préfères|Vous préférez] qu'on en parle à {reunion} ou avant ?",
        "[Sais-tu|Savez-vous] où en est {o} ?",
        "[Peux-tu|Pouvez-vous] me dire si {jour} [te|vous] convient pour {reunion} ?",
        "Une question : {o}, c'est pour quelle date déjà ?",
        "[Tu confirmes|Vous confirmez] [ta|votre] présence à {reunion} ?",
        "Combien de temps faut-il prévoir pour {o} ?",
        "[Tu as|Vous avez] une préférence pour {reunion} : présentiel ou visio ?",
        "J'aimerais avoir [ton|votre] avis sur {o} avant de répondre au client.",
        "Est-ce que {projet} est toujours prévu pour la fin du mois ?",
    ),
    "proposition de créneau": (
        "Nous sommes disponibles {jour} ou {jour2}, selon ce qui [t'arrange|vous arrange].",
        "Je peux [te|vous] proposer {jour} à {heure} ou {jour2} matin. [Dis-moi|Dites-moi] ce qui convient.",
        "Je suis libre {jour} après-midi si [tu veux|vous voulez] qu'on avance sur {o}.",
        "Quel créneau [t'arrange|vous arrange] pour {reunion} : {jour} ou {jour2} ?",
        "On pourrait caler {reunion} {jour} à {heure}, ça [te|vous] va ?",
        "Je [te|vous] laisse choisir entre {jour} et {jour2} pour {reunion}.",
        "Mes disponibilités pour {reunion} : {jour} matin, {jour2} toute la journée.",
    ),
    "invitation": (
        "[Je te|Je vous] propose {reunion} {jour} à {heure}, salle du 2e étage.",
        "Invitation à {reunion} : {jour} à {heure}, en visio.",
        "Merci de [bloquer|réserver] {jour} {heure} dans [ton|votre] agenda pour {reunion}.",
        "Ordre du jour de {reunion} ({jour} {heure}) : point sur {projet}, questions diverses.",
    ),
    "information": (
        "Pour information, {o} est disponible dans le dossier partagé.",
        "[Je te|Je vous] transmets {o} pour information.",
        "Veuillez trouver ci-joint {o}.",
        "Voici {o}, pour [ton|votre] information.",
        "Petit point d'info : {projet} avance bien, rien à signaler.",
        "Nouvel horaire pour {reunion} : {jour} à {heure}.",
        "Je serai en congé du {jour} au {jour2}, {tiers} assure l'intérim.",
        "La maintenance du serveur aura lieu {jour} soir. Aucune action n'est requise.",
        "Le compte rendu est en ligne ({reunion}).",
        "Je [te|vous] mets en copie de mon échange avec {tiers} concernant {o}, pour information.",
        "Nous avons mis en ligne {o} sur l'intranet.",
        "Pour [ta|votre] culture, la presse locale a parlé de {projet}.",
        "Rappel : {reunion} a lieu {jour} à {heure}.",
        "Les chiffres du mois sont stables, rien de particulier.",
    ),
    "confirmation": (
        "C'est fait : j'ai envoyé {o} au client.",
        "J'ai bien reçu {o}, merci.",
        "{O} : c'est validé de notre côté.",
        "La commande est confirmée pour {jour}.",
        "Le paiement a été effectué ce matin.",
        "Tout est en ordre concernant {o}.",
        "J'ai mis à jour {o} comme convenu.",
        "{tiers} a déjà signé {o}, rien à faire de [ton|votre] côté.",
        "Bonne nouvelle : le client a accepté {o}.",
        "Le problème est réglé, tout fonctionne de nouveau.",
    ),
    "remerciement": (
        "Merci pour [ton|votre] retour rapide !",
        "Merci, bien reçu.",
        "Super, merci beaucoup.",
        "Merci pour [ton|votre] aide sur {projet}, c'était parfait.",
        "Bravo pour {projet}, belle réussite collective.",
        "Parfait, merci !",
        "Merci à toute l'équipe pour l'organisation de la journée.",
        "Top, merci pour la réactivité.",
    ),
    "annulation": (
        "Finalement, inutile de {v} {o} : c'est réglé.",
        "[Oublie|Oubliez] ma demande, {tiers} s'en est occupé.",
        "Pas la peine de {v} {o}, c'est déjà fait.",
    ),
    "sans réponse": (
        "Pas besoin de répondre.",
        "Inutile de [me répondre|me faire un retour], c'est pour information.",
        "Sans nouvelles de [ta|votre] part, je considère que c'est bon.",
    ),
}

# Quelques formulations en anglais (environ 8 % des mails), mêmes étiquettes.
ENGLISH: dict[str, tuple[str, ...]] = {
    "demande explicite": (
        "Could you please review {oe} and send me your comments {quande}?",
        "Please sign {oe} {quande}.",
        "Can you take care of {oe}?",
    ),
    "question": ("Did the client receive {oe}?", "Are you available on {daye} for a quick call?"),
    "information": ("FYI, {oe} is attached. No action needed.", "Just a heads-up: {oe} is online."),
    "engagement de l'expéditeur": ("I'll look into it and get back to you {quande}.",),
    "remerciement": ("Thanks a lot, well received!",),
}

DEADLINES = (
    "avant {jour}",
    "d'ici {jour} midi",
    "avant la fin de la semaine",
    "aujourd'hui si possible",
    "dès que possible",
    "",
    "",
    "",
    "pour {jour}",
    "au plus tard {jour}",
    "avant {reunion}",
    "cette semaine",
)
SENDER_DELAYS = ("demain", "dans la journée", "d'ici ce soir", "en fin de semaine", "lundi", "")
DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi")
HOURS = ("9 h", "10 h 30", "11 h", "14 h", "15 h 30", "17 h")
GREETINGS_TU = ("Bonjour {prenom},", "Salut {prenom},", "Hello {prenom},", "Bonjour,", "")
GREETINGS_VOUS = ("Bonjour {prenom},", "Bonjour,", "Madame, Monsieur,", "")
OPENERS = (
    "Suite à notre échange de {jour},",
    "Comme évoqué ce matin,",
    "Je reviens sur {projet}.",
    "Petit point sur {projet} :",
    "J'espère que tout va bien.",
    "Pour faire suite à [ton|votre] message,",
)
CLOSINGS_REQUEST = ("Merci d'avance.", "Merci !", "Bonne journée,", "Cordialement,", "")
CLOSINGS = ("Bonne journée,", "Cordialement,", "Bien à vous,", "À bientôt,", "Belle journée !", "")
LEGAL = (
    "Ce message et ses pièces jointes sont confidentiels et destinés exclusivement à leur "
    "destinataire. Si vous l'avez reçu par erreur, merci de le supprimer."
)
