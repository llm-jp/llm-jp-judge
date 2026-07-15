import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class CultureDatasetItemMixin(BaseModel):
    """Mixin class for culture dataset item.

    Attributes:
        reference: Reference answer for each turn.
    """

    reference: list[str]


class CultureDatasetItem(DatasetItem, CultureDatasetItemMixin):
    """Dataset item for culture dataset."""

    pass


class CultureDatasetItemForEvaluation(DatasetItemForEvaluation, CultureDatasetItemMixin):
    """Dataset item for culture dataset for evaluation."""

    pass


def load_culture(path: str) -> list[CultureDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            item = CultureDatasetItem(ID=d["ID"], prompt=[d["text"]], reference=[d["output"]])
            data.append(item)

    return data


def load_culture_raw_output(path: str) -> list[CultureDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = CultureDatasetItem(**d)
            data.append(item)

    return data
