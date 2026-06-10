import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class QualityDatasetItemMixin(BaseModel):
    """Mixin class for quality dataset item.

    Attributes:
        text: Original text for each turn.
    """

    text: list[str]


class QualityDatasetItem(DatasetItem, QualityDatasetItemMixin):
    """Dataset item for quality dataset."""

    pass


class QualityDatasetItemForEvaluation(DatasetItemForEvaluation, QualityDatasetItemMixin):
    """Dataset item for quality dataset for evaluation."""

    pass


def load_quality(path: str) -> list[QualityDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            item = QualityDatasetItem(ID=d["ID"], prompt=[d["text"]], text=[d["text"]])
            data.append(item)

    return data


def load_quality_raw_output(path: str) -> list[QualityDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = QualityDatasetItem(**d)
            data.append(item)

    return data
