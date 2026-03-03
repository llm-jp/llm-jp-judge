import json
import logging
import re

from omegaconf import DictConfig

from src.llm_jp_judge.client.base import BaseClient
from src.llm_jp_judge.dashboard.base import BaseDashboard
from src.llm_jp_judge.dataset import DatasetItemForEvaluation


class BaseScoreExtractor:
    def __init__(self, regex: str):
        self.regex = regex

    def __call__(self, text: str) -> str:
        m = re.search(self.regex, text)

        if m is None:
            raise ValueError("No score found in the response")

        return m.group(1)


class BaseEvaluator:
    def __init__(
        self,
        client: BaseClient,
        dashboard: BaseDashboard,
        metadata: dict[str, str] | None = None,
        name: str = "base",
        use_reference: bool = False,
        system_prompt: str | None = None,
        sampling_params: dict[str, int | float | None] | DictConfig | None = None,
    ):
        if metadata is None:
            metadata = {}
        if sampling_params is None:
            sampling_params = {}

        self.client = client
        self.dashboard = dashboard
        self.name = name
        self.metadata = metadata
        self.use_reference = use_reference
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params

    def log_raw_outputs(self, raw_outputs: list[DatasetItemForEvaluation]):
        if self.dashboard is None:
            return

        columns = [
            "id",
            "metric",
            "evaluation prompt",
            "evaluation response",
            "score",
            "generate errors",
            "evaluation errors",
        ]
        data = [
            [
                score.ID,
                score.metric,
                score.prompt[0],
                score.response[0],
                score.pattern[0],
                json.dumps(score.generate_errors[0]),
                json.dumps(score.error_messages[0]),
            ]
            for score in raw_outputs
        ]
        self.dashboard.log_table(f"{self.name}_raw_output_table", columns=columns, data=data)

    def calc_error_rate(self, raw_outputs: list[DatasetItemForEvaluation]) -> tuple[float, float]:
        api_errors = [raw_output.response[0] is None for raw_output in raw_outputs]
        api_error_rate = sum(api_errors) / len(api_errors) * 100

        regex_match_errors = [raw_output.pattern[0] is None for raw_output in raw_outputs]
        regex_match_error_rate = sum(regex_match_errors) / len(regex_match_errors) * 100

        logging.info(f"API error rate: {api_error_rate:.2f}%")
        logging.info(f"Pattern match error rate: {regex_match_error_rate:.2f}%")

        return api_error_rate, regex_match_error_rate
