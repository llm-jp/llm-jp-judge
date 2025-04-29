import csv
import json
import hydra


def load_safety_boundary(path):
    path = hydra.utils.to_absolute_path(path)
    data = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, d in enumerate(reader):
            data.append(
                {
                    "ID": i,
                    "type": d["type"],
                    "safety": d["safety"],
                    "text": d["input"],
                    "prompt": d["input"],
                    "eval_aspect": d["eval_aspect"],
                    "ng_aspect": d["ng_aspect"],
                }
            )

    return data
