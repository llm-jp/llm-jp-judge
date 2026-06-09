import json
import logging
import re
from collections import defaultdict
from collections.abc import Sequence

from ..dataset.quality import QualityDatasetItem, QualityDatasetItemForEvaluation
from .base import BaseEvaluator, BaseScoreExtractor


class QualityScoreExtractor(BaseScoreExtractor):
    def __init__(self, regex: str, metrics: Sequence[str]):
        super().__init__(regex)
        self.metrics = metrics

    def __call__(self, text: str) -> dict[str, int]:  # type: ignore[override]
        scores = {}
        for metric, score in re.findall(self.regex, text):
            if metric in scores:
                raise ValueError("Duplicate metric")
            scores[metric] = int(score)

        if set(scores.keys()) != set(self.metrics):
            raise ValueError("Invalid score format")

        return scores


class QualityEvaluator(BaseEvaluator):
    def log_raw_outputs(self, raw_outputs: Sequence[QualityDatasetItemForEvaluation]):  # type: ignore[override]
        if self.dashboard is None:
            return

        metrics = self.prompt_template["metrics"]

        table = []
        header = [
            "id",
            "evaluation prompt",
            "evaluation response",
            *metrics,
            "generate errors",
            "evaluation errors",
        ]
        for raw_output in raw_outputs:
            if raw_output.pattern[0] is None:
                scores = {metric: None for metric in metrics}
            else:
                assert isinstance(raw_output.pattern[0], dict)
                scores = [raw_output.pattern[0].get(metric) for metric in metrics]
            table.append(
                [
                    raw_output.ID,
                    raw_output.prompt[0],
                    raw_output.response[0],
                    *scores,
                    json.dumps(raw_output.generate_errors[0], ensure_ascii=False),
                    json.dumps(raw_output.error_messages[0], ensure_ascii=False),
                ]
            )
        self.dashboard.log_table("quality_raw_output_table", columns=header, data=table)

    def __call__(self, responses: Sequence[QualityDatasetItem]) -> tuple[dict[str, float | None], dict[str, float]]:  # type: ignore[override]
        data: list[QualityDatasetItemForEvaluation] = []
        for res in responses:
            d = QualityDatasetItemForEvaluation(
                ID=res.ID,
                prompt=[self.prompt_template["prompt_template"].format(question=res.prompt, response=res.response)],
                text=res.text,
                generate_prompt=res.prompt,
                generate_response=res.response,
                generate_errors=res.error_messages,
            )
            data.append(d)

        score_extractor = QualityScoreExtractor(self.prompt_template["regex"], self.prompt_template["metrics"])
        raw_outputs = self.client(
            data,
            score_extractor=score_extractor,
            system_prompt=self.system_prompt,
            sampling_params=self.sampling_params,
        )

        scores = defaultdict(list)
        for raw_output in raw_outputs:
            if raw_output.pattern[0] is None:
                continue

            assert isinstance(raw_output.pattern[0], dict)
            for metric, score in raw_output.pattern[0].items():
                scores[metric].append(score)

        self.log_raw_outputs(raw_outputs)

        error_rates = {}
        (
            error_rates[f"{self.name}:api(%)"],
            error_rates[f"{self.name}:pattern_match(%)"],
        ) = self.calc_error_rate(raw_outputs)

        ave_scores = {
            f"{self.name}:{metric}": sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
