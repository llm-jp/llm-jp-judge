import json

import hydra

from src.llm_jp_judge.dataset import BaseModel, DatasetItem, DatasetItemForEvaluation


class SafetyDatasetItemMixin(BaseModel):
    text: list[str]
    reference: list[str]


class SafetyDatasetItem(DatasetItem, SafetyDatasetItemMixin):
    pass


class SafetyDatasetItemForEvaluation(DatasetItemForEvaluation, SafetyDatasetItemMixin):
    pass


def load_safety(path: str) -> list[SafetyDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            item = SafetyDatasetItem(ID=d["ID"], prompt=[d["text"]], text=[d["text"]], reference=[d["output"]])
            data.append(item)

    return data


def load_safety_raw_output(path: str) -> list[SafetyDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = SafetyDatasetItem(**d)
            data.append(item)

    return data
