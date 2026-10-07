# Journal des versions des contrats

Règle (guide §2.5) : versionnement semver.
- Ajout d'un champ **optionnel** → version mineure (1.0 → 1.1).
- Renommage, suppression, changement de type ou champ rendu obligatoire → version majeure (1.x → 2.0).
- Toute modification passe par une PR avec l'étiquette `contract-change` et 3 approbations de référents.

## [Non publié]

Propositions en attente du comité technique, rien n'est appliqué :
- ADR-0001 : champs optionnels `correlation_id` et `attributes` dans le contrat A (→ 1.1).
- ADR-0002 : la « source n°5 » de la preuve L1.1 réutilise `source = "file"` (aucun changement de contrat).

## Contrat A — Item 1.0 (2026-10-07)

- Version initiale, conforme au guide §2.2.
- Modèle pydantic v2 exécutable : `contracts/models/item.py` (champ inconnu refusé, objet immuable).
- Règles vérifiées en plus des types : identifiant préfixé par la source, URL en https, propriétaire présent dans `acl`, `updated_at` ≥ `created_at`, `start_at`/`end_at` obligatoires pour le calendrier, `completed` obligatoire pour les tâches, `is_read` réservé aux mails, extrait ≤ 500 caractères.
- JSON Schema exporté : `contracts/schemas/item.schema.json`.

## Contrat C — AuditEvent 1.0 (2026-10-07)

- Version initiale, conforme au guide §2.4.
- Interdit dans le journal : toute adresse e-mail ; `error_code` obligatoire si `status = "error"`.
- JSON Schema exporté : `contracts/schemas/audit_event.schema.json`.

## Contrat B — Brief

- Porté par le lot 2 (non inclus dans ce dossier).

## Fixtures

- `contracts/fixtures/valid/` : 3 exemples d'Item valides ; `contracts/fixtures/invalid/` : 21 cas rejetés (exigence L3.1 : au moins 15).
- `contracts/demo_data.py` : lecture des journées de démonstration pour le lot 2 (`load_items`, `available_days`).
- `contracts/tests/` : tests propres au dossier, sans dépendance aux lots.
- `contracts/fixtures/demo/` : 30 journées du manager `u_demo_001`, graine 42, générées par `uv run synth`. Toute régénération doit garder `uv run python -m contracts.validate_fixtures` au vert.
