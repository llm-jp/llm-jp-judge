from collections.abc import Sequence

from . import DatasetItem
from .mt_bench import load_mt_bench, load_mt_bench_raw_output
from .quality import load_quality, load_quality_raw_output
from .safety import load_safety, load_safety_raw_output


def load_dataset(name: str, path: str, size: int | None = None) -> Sequence[DatasetItem]:
    dataset: Sequence[DatasetItem]
    if name == "quality":
        dataset = load_quality(path)
    elif name == "safety":
        dataset = load_safety(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset = load_mt_bench(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if size is None:
        return dataset

    return dataset[:size]


def load_raw_output(name: str, path: str) -> Sequence[DatasetItem]:
    dataset: Sequence[DatasetItem]
    if name == "quality":
        dataset = load_quality_raw_output(path)
    elif name == "safety":
        dataset = load_safety_raw_output(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset = load_mt_bench_raw_output(path)
    else:
        raise ValueError(f"Unknown dataset for raw output: {name}")

    return dataset
