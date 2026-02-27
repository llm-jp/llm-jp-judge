import json

import hydra


def load_mt_bench(path: str) -> list[dict[str, str | list[str]]]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data.append(
                {
                    "ID": d["question_id"],  # string
                    "category": d["category"],  # string
                    "prompt": d["turns"],  # list of strings
                }
            )
    return data
