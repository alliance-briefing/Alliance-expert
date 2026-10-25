"""Cinq secteurs fictifs : qui écrit au manager, et de quoi on parle.

Pas d'expertise automobile : la démo du lot 1 la couvre, elle reste un secteur jamais vu
pour le test de transfert. Domaines en `.test` (réservés, RFC 2606), personnes inventées.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sector:
    key: str
    company: str
    domain: str
    manager: tuple[str, str]  # (prénom, fonction)
    boss: tuple[str, str]  # N+1 : (nom complet, fonction)
    peers: tuple[str, ...]  # pairs de direction
    team: tuple[str, ...]  # collègues et équipe
    clients: tuple[tuple[str, str], ...]  # (nom, organisation)
    suppliers: tuple[tuple[str, str], ...]
    objects: tuple[tuple[str, tuple[str, ...]], ...]  # (groupe nominal, verbes possibles)
    projects: tuple[str, ...]
    meetings: tuple[str, ...]


SECTORS = (
    Sector(
        key="conseil",
        company="Altiris Conseil",
        domain="altiris-conseil.test",
        manager=("Nora", "Directrice de mission"),
        boss=("Philippe Garnier", "Associé gérant"),
        peers=("Claire Fontaine", "Marc Delattre"),
        team=("Julien Perrot", "Sarah Benali", "Hugo Martin", "Léa Rousseau", "Yanis Cohen"),
        clients=(
            ("Isabelle Moreau", "Groupe Hélios"),
            ("Thierry Vasseur", "Banque Ondine"),
            ("Amina Diallo", "Mutuelle Atlas"),
        ),
        suppliers=(("Kevin Laurent", "DataPro Services"), ("Sophie Arnaud", "Traiteur Sésame")),
        objects=(
            ("la proposition commerciale pour Hélios", ("relire", "valider", "envoyer")),
            ("le rapport d'étape", ("relire", "valider", "compléter")),
            ("le planning de la mission", ("valider", "mettre à jour", "relire")),
            ("la facture de septembre", ("valider", "régler", "vérifier")),
            ("le compte rendu du comité de pilotage", ("relire", "compléter", "diffuser")),
            ("les feuilles de temps de l'équipe", ("valider", "vérifier")),
            ("le support de restitution", ("relire", "compléter", "valider")),
            ("l'avenant au contrat", ("signer", "relire", "valider")),
        ),
        projects=("la mission Hélios", "l'audit Ondine", "le diagnostic Atlas"),
        meetings=("le comité de pilotage", "le point d'avancement", "l'atelier de cadrage"),
    ),
    Sector(
        key="btp",
        company="Bâtirest",
        domain="batirest.test",
        manager=("Karim", "Conducteur de travaux principal"),
        boss=("Gilles Arnaud", "Directeur d'agence"),
        peers=("Valérie Lemoine", "Patrick Henry"),
        team=("Mehdi Saïdi", "Lucas Bernard", "Émilie Gauthier", "Thomas Roux", "Inès Faure"),
        clients=(
            ("Bernard Collin", "Mairie de Saint-Ouen-les-Prés"),
            ("Nathalie Petit", "Office Habitat 38"),
            ("Damien Royer", "SCI Les Tilleuls"),
        ),
        suppliers=(("Fabrice Marchal", "Béton Rhône"), ("Céline Weber", "Échafaudages Weber")),
        objects=(
            ("le devis du lot plomberie", ("valider", "signer", "vérifier")),
            ("le planning de chantier", ("valider", "mettre à jour", "relire")),
            ("le PV de réception", ("signer", "relire", "valider")),
            ("la commande de béton", ("valider", "confirmer", "passer")),
            ("le plan de prévention", ("signer", "relire", "compléter")),
            ("la situation de travaux n° 4", ("valider", "vérifier", "transmettre")),
            ("le rapport du bureau de contrôle", ("relire", "transmettre")),
            ("l'ordre de service", ("signer", "valider")),
        ),
        projects=(
            "le chantier de l'école Jean-Moulin",
            "la résidence Les Tilleuls",
            "la rénovation du gymnase",
        ),
        meetings=(
            "la réunion de chantier",
            "la visite de sécurité",
            "le point avec le maître d'ouvrage",
        ),
    ),
    Sector(
        key="clinique",
        company="Clinique du Parc",
        domain="clinique-du-parc.test",
        manager=("Agnès", "Cadre de santé"),
        boss=("François Léger", "Directeur des soins"),
        peers=("Muriel Blanc", "Olivier Mercier"),
        team=("Camille Durand", "Samir Haddad", "Julie Lefort", "Antoine Masson", "Élodie Brun"),
        clients=(("Martine Leroy", "ARS Auvergne"), ("Paul Girard", "Mutuelle Santé Plus")),
        suppliers=(("Damien Vidal", "MédiStock"), ("Laura Simon", "Blanchisserie Alpine")),
        objects=(
            ("le planning des gardes de novembre", ("valider", "relire", "mettre à jour")),
            ("la commande de matériel de soins", ("valider", "vérifier", "passer")),
            ("la fiche d'événement indésirable", ("relire", "compléter", "transmettre")),
            ("le protocole d'hygiène révisé", ("relire", "valider", "diffuser")),
            ("les demandes de congés de l'équipe", ("valider", "vérifier")),
            ("le compte rendu de la CME", ("relire", "compléter")),
            ("le plan de formation 2027", ("valider", "compléter", "relire")),
            ("la convention de stage", ("signer", "relire")),
        ),
        projects=(
            "la certification HAS",
            "l'ouverture de l'unité ambulatoire",
            "le déménagement de la pharmacie",
        ),
        meetings=("le staff du lundi", "la réunion des cadres", "la commission qualité"),
    ),
    Sector(
        key="ecommerce",
        company="Maison Lumen",
        domain="maison-lumen.test",
        manager=("Élise", "Responsable des opérations"),
        boss=("Raphaël Dumas", "Directeur général"),
        peers=("Nina Ferreira", "Bastien Lacroix"),
        team=("Chloé Marin", "Adrien Lopez", "Manon Picard", "Rayan Boucher", "Zoé Hamel"),
        clients=(("Laurent Mallet", "Galeries Orion"), ("Fatou Ndiaye", "Concept Store Nord")),
        suppliers=(
            ("Victor Renaud", "LogiTrans"),
            ("Aurore Chevalier", "Cartonnages du Sud"),
            ("Ivan Petrov", "Pixel Agency"),
        ),
        objects=(
            ("le bon de commande fournisseur", ("valider", "signer", "vérifier")),
            ("la campagne de soldes", ("valider", "relire", "lancer")),
            ("le tableau des retours clients", ("vérifier", "mettre à jour", "relire")),
            ("le contrat de transport", ("signer", "relire", "renégocier")),
            ("la fiche produit de la nouvelle gamme", ("relire", "valider", "compléter")),
            ("le prévisionnel de stock", ("valider", "mettre à jour", "vérifier")),
            ("le remboursement du client Mallet", ("valider", "traiter")),
            ("la maquette de la page d'accueil", ("relire", "valider")),
        ),
        projects=("le lancement de la gamme Aurore", "la migration du site", "les soldes d'hiver"),
        meetings=("le point hebdo", "la revue des ventes", "le comité produit"),
    ),
    Sector(
        key="collectivite",
        company="Ville de Valbonne-sur-Loire",
        domain="valbonne.test",
        manager=("Sandrine", "Directrice des services techniques"),
        boss=("Jean-Marc Ollivier", "Directeur général des services"),
        peers=("Hélène Berthier", "Christophe Navarro"),
        team=("Mathis Leblanc", "Aïcha Kone", "Romain Vallet", "Justine Morel", "Nicolas Prévost"),
        clients=(
            ("Mme Lucienne Perret", "Association des riverains"),
            ("Frédéric Aubry", "Communauté de communes"),
        ),
        suppliers=(("Sylvain Carré", "Espaces Verts Carré"), ("Delphine Ruiz", "Éclairage Ruiz")),
        objects=(
            ("la délibération sur les travaux de voirie", ("relire", "valider", "transmettre")),
            ("le marché d'éclairage public", ("valider", "signer", "relire")),
            ("le budget primitif des services techniques", ("valider", "relire", "arbitrer")),
            ("l'arrêté de circulation", ("signer", "relire", "valider")),
            ("le rapport pour le conseil municipal", ("relire", "compléter", "valider")),
            ("la réponse à l'association des riverains", ("relire", "valider", "envoyer")),
            ("le planning d'astreinte", ("valider", "mettre à jour")),
            ("le bon de commande des illuminations", ("signer", "valider")),
        ),
        projects=("la réfection de la place du marché", "le plan vélo", "la rénovation de l'école"),
        meetings=("la commission travaux", "le comité de direction", "la réunion de quartier"),
    ),
)
SECTOR_KEYS = tuple(s.key for s in SECTORS)
