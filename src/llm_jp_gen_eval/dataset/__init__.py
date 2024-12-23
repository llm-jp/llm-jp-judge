from .answer_carefully import load_answer_carefully
from .ichikara import load_ichikara
from .mt_bench import load_mt_bench


def load_dataset(name, path, size=None):
    if name == "ichikara":
        dataset = load_ichikara(path)
    elif name == "answer_carefully":
        dataset = load_answer_carefully(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset = load_mt_bench(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if size is None:
        return dataset

    return dataset[:size]
