import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class MTBenchDatasetItemMixin(BaseModel):
    """Mixin class for MT-Bench dataset item."""

    category: str


class MTBenchDatasetItem(DatasetItem, MTBenchDatasetItemMixin):
    """Dataset item for MT-Bench dataset."""

    pass


class MTBenchDatasetItemForEvaluation(DatasetItemForEvaluation, MTBenchDatasetItemMixin):
    """Dataset item for MT-Bench dataset for evaluation.

    Attributes:
        turn: Turn number.
        use_reference: Whether to use reference answer for evaluation.
        system_prompt: System prompt for the evaluation.
    """

    turn: int
    use_reference: bool
    system_prompt: str


def load_mt_bench(path: str) -> list[MTBenchDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = MTBenchDatasetItem(ID=d["question_id"], prompt=d["turns"], category=d["category"])
            data.append(item)

    return data


def load_mt_bench_raw_output(path: str) -> list[MTBenchDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = MTBenchDatasetItem(**d)
            data.append(item)

    return data
