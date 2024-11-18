import os
import hydra

from .client import load_client
from .utils.data import load_jsonl, save_jsonl

import logging


@hydra.main(config_path="./config", config_name="inference")
def main(cfg):
    output_path = hydra.utils.to_absolute_path(cfg.output.path)
    if os.path.exists(output_path) and not cfg.output.overwrite:
        logging.warning(f"Skipping inference, output file exists: {output_path}")
        logging.warning("Use output.overwrite=true to force re-run")
        return

    data = load_jsonl(cfg.input.path)
    inference_client = load_client(**cfg.client)
    for d in data:
        d["prompt"] = d["text"]
    responses = inference_client(data, system_prompt=cfg.system_prompt)

    success = [res["response"] is not None for res in responses]
    success_rate = sum(success) / len(success) * 100
    logging.info(f"Inference success rate: {success_rate:.2f}%")

    save_jsonl(cfg.output.path, responses)


if __name__ == "__main__":
    main()
