import logging
import re
from collections import defaultdict
from collections.abc import Sequence

from ..dataset.multichallenge import MultiChallengeDatasetItem, MultiChallengeDatasetItemForEvaluation
from .base import BaseEvaluator, BaseScoreExtractor


AXIS_NAME_MAPPING = {
    "INSTRUCTION_RETENTION": "指示保持",
    "INFERENCE_MEMORY": "記憶推論",
    "RELIABLE_VERSION_EDITING": "版編集",
    "SELF_COHERENCE": "自己一貫性",
}


class MultiChallengeScoreExtractor(BaseScoreExtractor):
    def __call__(self, text: str) -> str:
        matched = re.findall(self.regex, text)
        if not matched:
            raise ValueError("No score found in the response")

        parsed_label = matched[-1]
        return parsed_label


class MultiChallengeEvaluator(BaseEvaluator):
    def __call__(
        self, responses: Sequence[MultiChallengeDatasetItem]
    ) -> tuple[dict[str, float | None], dict[str, float]]:  # type: ignore[override]
        data: list[MultiChallengeDatasetItemForEvaluation] = []
        for res in responses:
            prompt = self.prompt_template["prompt_template"].format(
                response=res.response[-1], target_question=res.target_question
            )
            d = MultiChallengeDatasetItemForEvaluation(
                ID=res.ID,
                prompt=[prompt],
                axis=res.axis,
                target_question=res.target_question,
                generate_prompt=res.prompt,
                generate_response=res.response,
                generate_errors=res.error_messages,
            )
            data.append(d)

        score_extractor = MultiChallengeScoreExtractor(regex=self.prompt_template["regex"])
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

            assert isinstance(raw_output.pattern[0], str)
            score = float(raw_output.pattern[0] == "YES")
            scores["全体スコア"].append(score)
            scores[AXIS_NAME_MAPPING[raw_output.axis]].append(score)

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
