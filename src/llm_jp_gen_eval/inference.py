import os
import hydra

from .client import load_client
from .utils.data import save_jsonl, save_json
from .dataset import load_dataset

import logging


def inference(cfg, client, benchmark_cfg):
    output_dir = hydra.utils.to_absolute_path(cfg.output.dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{benchmark_cfg.name}.jsonl")

    if cfg.output.overwrite and os.path.exists(output_path):
        logging.info(f"Skipping inference for {benchmark_cfg.name} as output exists")
        return

    logging.info(f"Loading dataset: {benchmark_cfg.name}")
    data = load_dataset(benchmark_cfg.name, benchmark_cfg.dataset.path)

    logging.info(f"Running inference on {len(data)} samples")
    responses = client(data, system_prompt=cfg.system_prompt)

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


@hydra.main(config_path="./config", config_name="inference")
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

        logging.info(f"Running inference on benchmark: {benchmark_cfg.name}")
        inference(cfg, client, benchmark_cfg)

    save_metadata(cfg)


if __name__ == "__main__":
    main()
