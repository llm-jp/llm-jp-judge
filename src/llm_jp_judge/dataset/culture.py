import json
import hydra


def load_culture(path):
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            data.append(
                {
                    "ID": d["ID"],
                    "prompt": d["text"],
                    "reference": d["output"],
                }
            )
    return data
