import json

import hydra

from src.llm_jp_judge.dataset import BaseModel, DatasetItem, DatasetItemForEvaluation


class MTBenchDatasetItemMixin(BaseModel):
    category: str


class MTBenchDatasetItem(DatasetItem, MTBenchDatasetItemMixin):
    pass


class MTBenchDatasetItemForEvaluation(DatasetItemForEvaluation, MTBenchDatasetItemMixin):
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
