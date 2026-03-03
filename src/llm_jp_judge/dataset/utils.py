from src.llm_jp_judge.dataset import DatasetItem
from src.llm_jp_judge.dataset.mt_bench import MTBenchDatasetItem, load_mt_bench, load_mt_bench_raw_output
from src.llm_jp_judge.dataset.quality import QualityDatasetItem, load_quality, load_quality_raw_output
from src.llm_jp_judge.dataset.safety import SafetyDatasetItem, load_safety, load_safety_raw_output


def load_dataset(name: str, path: str, size: int | None = None) -> list[DatasetItem]:
    if name == "quality":
        dataset: list[QualityDatasetItem] = load_quality(path)
    elif name == "safety":
        dataset: list[SafetyDatasetItem] = load_safety(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset: list[MTBenchDatasetItem] = load_mt_bench(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if size is None:
        return dataset

    return dataset[:size]


def load_raw_output(name: str, path: str) -> list[DatasetItem]:
    if name == "quality":
        dataset: list[QualityDatasetItem] = load_quality_raw_output(path)
    elif name == "safety":
        dataset: list[SafetyDatasetItem] = load_safety_raw_output(path)
    elif name in ["mt_bench", "ja_mt_bench"]:
        dataset: list[MTBenchDatasetItem] = load_mt_bench_raw_output(path)
    else:
        raise ValueError(f"Unknown dataset for raw output: {name}")

    return dataset
