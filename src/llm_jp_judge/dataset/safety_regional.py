import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class SafetyRegionalDatasetItemMixin(BaseModel):
    """Mixin class for safety dataset item.

    Attributes:
        text: Original text for each turn.
        reference: Reference answer for each turn.
    """

    text: list[str]
    reference: list[str]


class SafetyRegionalDatasetItem(DatasetItem, SafetyRegionalDatasetItemMixin):
    """Dataset item for safety dataset."""

    pass


class SafetyRegionalDatasetItemForEvaluation(DatasetItemForEvaluation, SafetyRegionalDatasetItemMixin):
    """Dataset item for safety dataset for evaluation."""

    pass


def load_safety_regional(path: str) -> list[SafetyRegionalDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            item = SafetyRegionalDatasetItem(
                ID=d["ID"], prompt=[d["text"]], text=[d["text"]], reference=[d["output-sample"]]
            )
            data.append(item)

    return data


def load_safety_regional_raw_output(path: str) -> list[SafetyRegionalDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = SafetyRegionalDatasetItem(**d)
            data.append(item)

    return data
