import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class SafetyBorderlineDatasetItemMixin(BaseModel):
    text: list[str]
    reference: list[str]


class SafetyBorderlineDatasetItem(DatasetItem, SafetyBorderlineDatasetItemMixin):
    pass


class SafetyBorderlineDatasetItemForEvaluation(DatasetItemForEvaluation, SafetyBorderlineDatasetItemMixin):
    pass


def load_safety_boarderline(path: str) -> list[SafetyBorderlineDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            item = SafetyBorderlineDatasetItem(
                ID=d["ID"], prompt=[d["text"]], text=[d["text"]], reference=[d["output"]]
            )
            data.append(item)

    return data


def load_safety_boarderline_raw_output(path: str) -> list[SafetyBorderlineDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = SafetyBorderlineDatasetItem(**d)
            data.append(item)

    return data
