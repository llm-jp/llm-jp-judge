import os
import hydra

from collections import defaultdict

from omegaconf import OmegaConf

from .client import load_client
from .utils.data import save_jsonl, save_json
from .dataset import load_dataset

import logging


def generate(cfg, client, benchmark_cfg):
    output_dir = hydra.utils.to_absolute_path(cfg.output.dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_cfg.name}.jsonl")

    if not cfg.output.overwrite and os.path.exists(output_path):
        logging.info(f"Skipping generate for {benchmark_cfg.name} as output exists")
        return

    logging.info(f"Loading dataset: {benchmark_cfg.name}")
    data = load_dataset(
        benchmark_cfg.name, benchmark_cfg.dataset.path, benchmark_cfg.dataset.size
    )

    if (
        "category_sampling_params" in benchmark_cfg
    ):  # データカテゴリー毎にサンプリングパラメータを設定する場合: MT-Bench用
        category_sampling_params = OmegaConf.to_container(
            benchmark_cfg.category_sampling_params, resolve=True
        )

        categorized_data = defaultdict(list)
        for i, d in enumerate(data):
            d["original_index"] = i
            categorized_data[d["category"]].append(d)

        responses = []
        for category, category_data in categorized_data.items():
            sampling_params = OmegaConf.to_container(
                benchmark_cfg.sampling_params, resolve=True
            )
            sampling_params.update(category_sampling_params.get(category, {}))

            logging.info(
                f"Running generate on {len(category_data)} samples in category: {category}"
            )
            responses += client(
                category_data,
                system_prompt=benchmark_cfg.system_prompt,
                sampling_params=sampling_params,
            )
        responses = sorted(responses, key=lambda x: x["original_index"])
        for res in responses:
            del res["original_index"]

    else:
        logging.info(f"Running generate on {len(data)} samples")
        responses = client(
            data,
            system_prompt=benchmark_cfg.system_prompt,
            sampling_params=benchmark_cfg.sampling_params,
        )

    success = [res["response"] is not None for res in responses]
    success_rate = sum(success) / len(success) * 100
    logging.info(f"Inference success rate: {success_rate:.2f}%")

    logging.info(f"Saving responses to {output_path}")
    save_jsonl(output_path, responses)


def save_metadata(cfg):
    output_dir = hydra.utils.to_absolute_path(cfg.output.dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metadata.json")

    logging.info(f"Saving metadata to {output_path}")
    metadata = {
        "model_name": cfg.client.model_name,
    }
    save_json(output_path, metadata)


@hydra.main(config_path="./config", config_name="generate")
def main(cfg):
    any_specified = any(
        benchmark_cfg.dataset.path for benchmark_cfg in cfg.benchmark.values()
    )
    if not any_specified:
        logging.error("Must specify at least one dataset.path")
        return

    logging.info(f"Loading client: {cfg.client.model_name}")
    client = load_client(**cfg.client)

    for benchmark_cfg in cfg.benchmark.values():
        if not benchmark_cfg.dataset.path:
            continue

        logging.info(f"Running generate on benchmark: {benchmark_cfg.name}")
        generate(cfg, client, benchmark_cfg)

    save_metadata(cfg)


if __name__ == "__main__":
    main()
