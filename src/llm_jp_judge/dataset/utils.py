from collections.abc import Sequence

from . import DatasetItem
from .culture import load_culture, load_culture_raw_output
from .mt_bench import load_mt_bench, load_mt_bench_raw_output
from .quality import load_quality, load_quality_raw_output
from .safety import load_safety, load_safety_raw_output
from .safety_boundary import load_safety_boundary, load_safety_boundary_raw_output


def load_dataset(name: str, path: str, size: int | None = None) -> Sequence[DatasetItem]:
    dataset: Sequence[DatasetItem]
    if name == "quality_ja":
        dataset = load_quality(path)
    elif name == "safety_ja":
        dataset = load_safety(path)
    elif name == "culture_ja":
        dataset = load_culture(path)
    elif name == "safety_boundary_ja":
        dataset = load_safety_boundary(path)
    elif name in ["mt_bench_en", "mt_bench_ja"]:
        dataset = load_mt_bench(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if size is None:
        return dataset

    return dataset[:size]


def load_raw_output(name: str, path: str) -> Sequence[DatasetItem]:
    dataset: Sequence[DatasetItem]
    if name == "quality_ja":
        dataset = load_quality_raw_output(path)
    elif name == "safety_ja":
        dataset = load_safety_raw_output(path)
    elif name == "culture_ja":
        dataset = load_culture_raw_output(path)
    elif name == "safety_boundary_ja":
        dataset = load_safety_boundary_raw_output(path)
    elif name in ["mt_bench_en", "mt_bench_ja"]:
        dataset = load_mt_bench_raw_output(path)
    else:
        raise ValueError(f"Unknown dataset for raw output: {name}")

    return dataset
