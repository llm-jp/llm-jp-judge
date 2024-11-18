import os
import json

import hydra


def load_jsonl(path):
    path = hydra.utils.to_absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def save_jsonl(path, data):
    path = hydra.utils.to_absolute_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return
