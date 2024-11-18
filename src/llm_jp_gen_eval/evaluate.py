import re
import os

import hydra
import logging

from .client import load_client
from .evaluator import load_evaluator
from .utils.data import load_jsonl


@hydra.main(config_path="./config", config_name="evaluate")
def main(cfg):
    logging.info(f"Loading client: {cfg.client.model_name}")
    client = load_client(**cfg.client)

    data = load_jsonl(cfg.input.path)

    evaluator = load_evaluator(client, **cfg.aspect)
    evaluator(data)


if __name__ == "__main__":
    main()
