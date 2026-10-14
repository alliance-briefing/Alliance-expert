# Ajouter une source (guide L1.1)

Une source = **un nouveau dossier** dans `packages/collect/connectors/`. Rien d'autre à modifier :
le registre la trouve tout seul et elle est active par défaut.

## Étapes

1. Créer `packages/collect/connectors/ma_source/` avec deux fichiers :
   - `__init__.py` : `from packages.collect.connectors.ma_source.connector import MaSourceConnector`
   - `connector.py` : une classe qui hérite de `Connector` (voir le modèle ci-dessous).
2. (Conseillé) ajouter ses tests dans `packages/collect/connectors/ma_source/test_ma_source.py`.
3. Vérifier : `uv run python -m packages.collect.cli --user u_demo_001 --day 2026-10-05`
   doit afficher une ligne `INFO connecteur ma_source : N Items`.

## Modèle

```python
from collections.abc import Iterator

from contracts.models import Item
from packages.collect.core.connector import Connector, Health, Window


class MaSourceConnector(Connector):
    name = "ma_source"  # unique, en minuscules
    version = "0.1.0"

    def required_scopes(self) -> list[str]:
        return []  # autorisations en lecture seule

    def collect(
        self, user: str, window: Window
    ) -> Iterator[Item]: ...  # yield Item(...) pour chaque élément de la période

    def health(self) -> Health:
        return Health(True, "ok")
```

## Règles

- `collect` ne renvoie que des `Item` valides (contrat A) appartenant à `user` ; le reste est rejeté.
- Si la source plante, les autres continuent : l'erreur apparaît dans les warnings.
- Ne jamais mettre de jeton, d'adresse e-mail ou de message d'erreur brut dans les journaux.
- Pour couper une source sans supprimer son dossier : `disabled = ["ma_source"]` dans `collect.toml`.

## Preuve « source n°5 »

La personne qui ajoute la source n°5 ne doit pas avoir écrit l'interface. Noter dans l'issue :
l'heure de début et de fin (moins de 4 h) et la liste des fichiers créés (tous dans le nouveau dossier).
