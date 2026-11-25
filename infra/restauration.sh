#!/usr/bin/env bash
# Restauration du volume de données depuis une archive de infra/sauvegarde.sh (L3.8).
# Usage : infra/restauration.sh ARCHIVE [-y]
# Remplace TOUTES les données du volume (items, journaux, briefs, retours) : l'API est
# arrêtée pendant l'opération, puis relancée. -y : sans demande de confirmation.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage : $0 ARCHIVE [-y]" >&2
  exit 2
fi
archive=$1
tar tzf "$archive" > /dev/null || { echo "archive illisible : $archive" >&2; exit 1; }
if [ "${2:-}" != "-y" ]; then
  read -r -p "Remplacer toutes les données du volume par $archive ? [o/N] " reponse
  [ "$reponse" = "o" ] || { echo "abandon"; exit 1; }
fi
compose=(docker compose -f "$(dirname "$0")/docker-compose.yml")
"${compose[@]}" stop api
"${compose[@]}" run --rm --no-deps -T api sh -c 'find /data -mindepth 1 -delete && tar xzf - -C /data' < "$archive"
"${compose[@]}" up -d --wait api
echo "restauré : $archive"
