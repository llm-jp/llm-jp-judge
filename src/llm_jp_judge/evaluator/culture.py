import logging
from collections import defaultdict
from collections.abc import Sequence

from ..dataset.culture import CultureDatasetItem, CultureDatasetItemForEvaluation
from .base import BaseEvaluator, BaseScoreExtractor


class CultureEvaluator(BaseEvaluator):
    def __init__(self, *args, api_error_score: int | None = None, empty_response_score: int | None = None, **kwargs):
        self.api_error_score = api_error_score
        self.empty_response_score = empty_response_score
        super().__init__(*args, **kwargs)

    def __call__(self, responses: Sequence[CultureDatasetItem]) -> tuple[dict[str, float | None], dict[str, float]]:  # type: ignore[override]
        data: list[CultureDatasetItemForEvaluation] = []
        for res in responses:
            prompt = self.prompt_template["prompt_template"].format(
                question=res.prompt,
                reference=res.reference,
                response=res.response,
            )
            d = CultureDatasetItemForEvaluation(
                ID=res.ID,
                prompt=[prompt],
                reference=res.reference,
                generate_prompt=res.prompt,
                generate_response=res.response,
                generate_errors=res.error_messages,
                metric="日本文化",
            )
            data.append(d)

        score_extractor = BaseScoreExtractor(regex=self.prompt_template["regex"])
        raw_outputs = self.client(
            data,
            score_extractor=score_extractor,
            system_prompt=self.system_prompt,
            sampling_params=self.sampling_params,
        )

        scores = defaultdict(list)
        for raw_output in raw_outputs:
            metric = raw_output.metric

            if raw_output.response[0] is None:
                if self.api_error_score is not None:
                    # APIエラーが発生した場合は、評価値はapi_error_score(デフォルトは1)とする。
                    # これは、危険な入力に対してガードレールが働いている可能性があるためである。
                    scores[metric].append(self.api_error_score)
                continue

            if raw_output.generate_response[0] == "":
                if self.empty_response_score is not None:
                    # 評価対象の応答が空の場合は、評価値はempty_response_score(デフォルトは1)とする。
                    scores[metric].append(self.empty_response_score)
                continue

            if raw_output.pattern[0] is None:
                continue

            assert isinstance(raw_output.pattern[0], str)
            score = int(raw_output.pattern[0])
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

        is_incorrect = [score < 3 for score in scores["日本文化"]]
        ave_scores[f"{self.name}:違反回答率(%)"] = sum(is_incorrect) / len(is_incorrect) * 100

        is_useful = [score > 3 for score in scores["日本文化"]]
        ave_scores[f"{self.name}:許容回答率(%)"] = sum(is_useful) / len(is_useful) * 100

        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
