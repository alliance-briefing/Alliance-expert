#!/usr/bin/env bash
# Sauvegarde du volume de données de l'application (L3.8) : archive tar.gz de /data.
# Usage : infra/sauvegarde.sh [ARCHIVE]   (défaut : sauvegardes/briefing-AAAAMMJJ-HHMMSS.tgz)
# L'archive sort du conteneur par la sortie standard : aucun montage de dossier, aucune
# image à télécharger. Sous Windows : Git Bash (pas PowerShell, qui abîme les flux binaires).
set -euo pipefail
compose=(docker compose -f "$(dirname "$0")/docker-compose.yml")
archive="${1:-sauvegardes/briefing-$(date +%Y%m%d-%H%M%S).tgz}"
mkdir -p "$(dirname "$archive")"
trap 'rm -f "$archive"; echo "sauvegarde en échec" >&2' ERR  # jamais d'archive tronquée
"${compose[@]}" run --rm --no-deps -T api tar czf - -C /data . > "$archive"
tar tzf "$archive" > /dev/null  # archive lisible, sinon échec
echo "sauvegarde : $archive ($(du -h "$archive" | cut -f1))"
