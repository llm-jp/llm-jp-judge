import csv
import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class SafetyBoundaryDatasetItemMixin(BaseModel):
    text: list[str]
    type: str
    safety: str
    eval_aspect: str
    ng_aspect: str


class SafetyBoundaryDatasetItem(DatasetItem, SafetyBoundaryDatasetItemMixin):
    pass


class SafetyBoundaryDatasetItemForEvaluation(DatasetItemForEvaluation, SafetyBoundaryDatasetItemMixin):
    pass


def load_safety_boundary(path: str) -> list[SafetyBoundaryDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, d in enumerate(reader):
            item = SafetyBoundaryDatasetItem(
                ID=i,
                prompt=[d["input"]],
                text=[d["input"]],
                type=d["type"],
                safety=d["safety"],
                eval_aspect=d["eval_aspect"],
                ng_aspect=d["ng_aspect"],
            )
            data.append(item)

    return data


def load_safety_boundary_raw_output(path: str) -> list[SafetyBoundaryDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = SafetyBoundaryDatasetItem(**d)
            data.append(item)

    return data
