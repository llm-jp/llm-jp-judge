import logging
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Sequence

import hydra
from omegaconf import DictConfig, OmegaConf

from .client import load_client
from .client.base import BaseClient
from .dataset import DatasetItem
from .dataset.mt_bench import MTBenchDatasetItem
from .dataset.utils import load_dataset
from .utils.data import load_json, save_json, save_jsonl


def generate(cfg: DictConfig, client: BaseClient, benchmark_cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output.dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_cfg.name}.jsonl")

    if not cfg.output.overwrite and os.path.exists(output_path):
        logging.info(f"Skipping generate for {benchmark_cfg.name} as output exists")
        return

    logging.info(f"Loading dataset: {benchmark_cfg.name}")
    data = load_dataset(benchmark_cfg.name, benchmark_cfg.dataset.path, benchmark_cfg.dataset.size)

    responses: Sequence[DatasetItem]
    if (
        "category_sampling_params" in benchmark_cfg
    ):  # データカテゴリー毎にサンプリングパラメータを設定する場合: MT-Bench用
        category_sampling_params = OmegaConf.to_container(benchmark_cfg.category_sampling_params, resolve=True)
        assert isinstance(category_sampling_params, dict)

        categorized_data: dict[str, list[DatasetItem]] = defaultdict(list)
        for i, d in enumerate(data):
            assert isinstance(d, MTBenchDatasetItem)
            d.original_index = i
            categorized_data[d.category].append(d)

        responses = []
        for category, category_data in categorized_data.items():
            sampling_params = OmegaConf.to_container(benchmark_cfg.sampling_params, resolve=True)
            assert isinstance(sampling_params, dict)
            sampling_params.update(category_sampling_params.get(category, {}))

            logging.info(f"Running generate on {len(category_data)} samples in category: {category}")
            responses += client(
                category_data,
                system_prompt=benchmark_cfg.system_prompt,
                sampling_params=sampling_params,
            )

        def sort_key(x: DatasetItem) -> int:
            assert x.original_index is not None
            return x.original_index

        responses = sorted(responses, key=sort_key)
        for res in responses:
            res.original_index = None

    else:
        logging.info(f"Running generate on {len(data)} samples")
        responses = client(
            data,
            system_prompt=benchmark_cfg.system_prompt,
            sampling_params=benchmark_cfg.sampling_params,
        )

    success = [all(response_text is not None for response_text in res.response) for res in responses]
    success_rate = sum(success) / len(success) * 100
    logging.info(f"Inference success rate: {success_rate:.2f}%")

    logging.info(f"Saving responses to {output_path}")
    save_jsonl(output_path, (res.model_dump(exclude={"original_index"}) for res in responses))


def save_metadata(cfg: DictConfig):
    output_dir = hydra.utils.to_absolute_path(cfg.output.dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metadata.json")

    logging.info(f"Saving metadata to {output_path}")
    metadata = {
        "model_name": cfg.client.model_name,
    }
    save_json(output_path, metadata)


def publish_model_switch(output: Path, staged: Path, existing: set[str]) -> None:
    # evaluate reads every top-level JSONL under one label. Withdraw the old set before switching labels.
    previous = Path(mkdtemp(prefix=".previous-model-", dir=output))
    metadata = output / "metadata.json"
    old_metadata = previous / metadata.name
    new_metadata = staged / metadata.name
    undo: list[tuple[Path, Path]] = []
    try:
        if metadata.exists():
            shutil.copy2(metadata, old_metadata)
        for name in sorted(existing):
            (output / name).replace(previous / name)
            undo.append((previous / name, output / name))

        new_metadata.replace(metadata)
        undo.append((old_metadata, metadata) if old_metadata.exists() else (metadata, new_metadata))
        for response in sorted(staged.glob("*.jsonl")):
            destination = output / response.name
            response.replace(destination)
            undo.append((destination, response))
    except BaseException:
        try:
            for source, destination in reversed(undo):
                source.replace(destination)
        except OSError as error:
            raise RuntimeError(f"Could not roll back model switch. Remaining originals are in {previous}") from error
        shutil.rmtree(previous)
        raise
    shutil.rmtree(previous)


def generate_benchmarks(cfg: DictConfig, benchmarks: list[DictConfig]) -> None:
    logging.info(f"Loading client: {cfg.client.model_name}")
    client = load_client(**cfg.client)
    for benchmark_cfg in benchmarks:
        logging.info(f"Running generate on benchmark: {benchmark_cfg.name}")
        generate(cfg, client, benchmark_cfg)


@hydra.main(config_path="./config", config_name="generate")
def main(cfg: DictConfig):
    benchmarks = [b for b in cfg.benchmark.values() if b.dataset.path]
    if not benchmarks:
        logging.error("Must specify at least one dataset.path")
        return

    output = Path(hydra.utils.to_absolute_path(cfg.output.dir))
    existing = set(glob("*.jsonl", root_dir=output))
    metadata = output / "metadata.json"
    owner = load_json(str(metadata)).get("model_name") if metadata.is_file() else None
    if not existing or (owner is not None and owner == cfg.client.model_name):
        if not existing:
            save_metadata(cfg)
        generate_benchmarks(cfg, benchmarks)
        if existing:
            save_metadata(cfg)
        return

    if not cfg.output.overwrite:
        raise ValueError(
            f"Cannot reuse responses in {output}: recorded model {owner!r}, requested {cfg.client.model_name!r}. "
            "Use a different output directory or regenerate with output.overwrite=true."
        )
    remaining = existing - {f"{b.name}.jsonl" for b in benchmarks}
    if remaining:
        raise ValueError(f"Select the remaining benchmarks to change models: {', '.join(sorted(remaining))}")

    with TemporaryDirectory(dir=output) as tmp:
        staged = deepcopy(cfg)
        staged.output.dir = tmp
        generate_benchmarks(staged, benchmarks)
        save_metadata(staged)
        publish_model_switch(output, Path(tmp), existing)


if __name__ == "__main__":
    main()
