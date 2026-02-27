import json

import hydra


def load_safety(path: str) -> list[dict[str, str]]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            data.append(
                {
                    "ID": d["ID"],  # string
                    "text": d["text"],  # string
                    "prompt": d["text"],  # string
                    "reference": d["output"],  # string
                }
            )
    return data
