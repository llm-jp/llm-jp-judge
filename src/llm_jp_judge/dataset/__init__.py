from llm_jp_judge.dataset.mt_bench import load_mt_bench
from llm_jp_judge.dataset.quality import load_quality
from llm_jp_judge.dataset.safety import load_safety


def load_dataset(name: str, path: str, size: int | None = None) -> list[dict[str, str | list[str]]]:
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
