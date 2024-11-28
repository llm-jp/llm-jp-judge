from .answer_carefully import load_answer_carefully
from .ichikara import load_ichikara


def load_dataset(name, path):
    if name == "ichikara":
        return load_ichikara(path)
    if name == "answer_carefully":
        return load_answer_carefully(path)
    raise ValueError(f"Unknown dataset: {name}")
