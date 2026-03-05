import json
import os
from collections.abc import Iterable
from typing import Any

import hydra


def load_json(path: str) -> Any:
    path = hydra.utils.to_absolute_path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_jsonl(path: str) -> list[Any]:
    path = hydra.utils.to_absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def save_json(path: str, data: Any):
    path = hydra.utils.to_absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def save_jsonl(path: str, data: Iterable[Any]):
    path = hydra.utils.to_absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
