import logging

from copy import deepcopy
from collections import defaultdict

from .base import BaseEvaluator, BaseScoreExtractor

PROMPT_TEMPLATE = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. The expected language is Japanese. Responses in languages other than Japanese will incur score deductions unless specifically required. Failure to use Japanese at all will result in the lowest evaluation. However, using Japanese is not mandatory when providing only Python scripts or calculation results, where Japanese is not essential. Additionally, your explanation of judgement should be in Japanese. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Reference Answer]
{reference}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]"""

SCORE_REGEX = r"\[\[(\d+)\]\]"


class CultureEvaluator(BaseEvaluator):
    def __init__(self, *args, api_error_score=None, **kwargs):
        self.api_error_score = api_error_score
        super().__init__(*args, **kwargs)

    def __call__(self, responses):
        data = []
        for res in responses:
            d = deepcopy(res)
            d["metric"] = "日本文化"
            d["generate_prompt"] = d["prompt"]
            d["generate_response"] = d["response"]
            d["generate_errors"] = d.get("error_messages", [])

            d["prompt"] = PROMPT_TEMPLATE.format(
                question=d["prompt"],
                reference=d["reference"],
                response=d["response"],
            )
            data.append(d)

        score_extractor = BaseScoreExtractor(regex=SCORE_REGEX)
        raw_outputs = self.client(
            data,
            score_extractor=score_extractor,
            system_prompt=self.system_prompt,
            sampling_params=self.sampling_params,
        )

        scores = defaultdict(list)
        for raw_output in raw_outputs:
            metric = raw_output["metric"]

            if raw_output.get("response") is None:
                if self.api_error_score is not None:
                    # APIエラーが発生した場合は、評価値はapi_error_score(デフォルトは1)とする。
                    # これは、危険な入力に対してガードレールが働いている可能性があるためである。
                    scores[metric].append(self.api_error_score)
                continue

            if raw_output.get("pattern") is None:
                continue
            score = int(raw_output["pattern"])
            scores[metric].append(score)

        self.log_raw_outputs(raw_outputs)

        error_rates = {}
        (
            error_rates[f"{self.name}:api(%)"],
            error_rates[f"{self.name}:pattern_match(%)"],
        ) = self.calc_error_rate(raw_outputs)

        ave_scores = {
            f"culture:{metric}": sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
