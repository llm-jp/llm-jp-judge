import re
import os
import glob

import hydra
import logging

from .client import load_client
from .evaluator import load_evaluator
from .utils.data import load_jsonl, load_json
from .dashboard import load_dashboard


def load_metadata(cfg):
    input_dir = hydra.utils.to_absolute_path(cfg.input.dir)
    metadata_path = os.path.join(input_dir, "metadata.json")
    assert os.path.exists(metadata_path), f"Metadata not found at {metadata_path}"
    return load_json(metadata_path)


def load_raw_outputs(cfg):
    input_dir = hydra.utils.to_absolute_path(cfg.input.dir)
    output_paths = glob.glob(os.path.join(input_dir, "*.jsonl"))

    raw_outputs = {}
    for output_path in output_paths:
        assert os.path.exists(output_path), f"Responses not found at {output_path}"

        benchmark_name = os.path.splitext(os.path.basename(output_path))[0]
        raw_outputs[benchmark_name] = load_jsonl(output_path)

    assert len(raw_outputs) > 0, f"No raw outputs (.jsonl) found in {cfg.input.dir}"
    return raw_outputs


@hydra.main(config_path="./config", config_name="evaluate")
def main(cfg):
    logging.info(f"Loading metadata")
    metadata = load_metadata(cfg)

    logging.info(f"Loading raw outputs")
    raw_outputs = load_raw_outputs(cfg)

    logging.info(f"Loading dashboard")
    dashboard = load_dashboard(cfg, **cfg.get("dashboard", {}))

    logging.info(f"Loading client: {cfg.client.model_name}")
    client = load_client(**cfg.client)

    all_scores, all_error_rates = {}, {}
    for benchmark_name, data in raw_outputs.items():
        logging.info(f"Evaluating benchmark: {benchmark_name}")
        benchmark_cfg = cfg.benchmark[benchmark_name]
        evaluator = load_evaluator(
            client, dashboard, metadata=metadata, **benchmark_cfg
        )
        scores, error_rates = evaluator(data)
        all_scores.update(scores)
        all_error_rates.update(error_rates)

    metrics = list(all_scores.keys())
    columns = ["generate model", "evaluation model"] + metrics
    row = [metadata["model_name"], cfg.client.model_name] + [
        all_scores[metric] for metric in metrics
    ]
    dashboard.log_table("score_table", columns=columns, data=[row])

    header = list(all_error_rates.keys())
    columns = ["generate model", "evaluation model"] + header
    row = [metadata["model_name"], cfg.client.model_name] + [
        all_error_rates[key] for key in header
    ]
    dashboard.log_table("evaluate_error_rate_table", columns=columns, data=[row])

    dashboard.close()


if __name__ == "__main__":
    main()
