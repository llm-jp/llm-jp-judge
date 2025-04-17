from .safety import load_safety
from .quality import load_quality
from .mt_bench import load_mt_bench
from .culture import load_culture


def load_dataset(name, path, size=None):
    if name == "quality":
        dataset = load_quality(path)
    elif name == "safety":
        dataset = load_safety(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset = load_mt_bench(path)
    elif name in ["culture"]:
        dataset = load_culture(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if size is None:
        return dataset

    return dataset[:size]
