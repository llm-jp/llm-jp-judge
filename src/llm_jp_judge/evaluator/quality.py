import json
import logging
import re
from collections import defaultdict

from src.llm_jp_judge.dataset.quality import QualityDatasetItem, QualityDatasetItemForEvaluation
from src.llm_jp_judge.evaluator.base import BaseEvaluator, BaseScoreExtractor


PROMPT_TEMPLATE = """[指示]
質問に対するAIアシスタントの回答を以下の基準で評価してください。

正確性: 応答が事実を述べているか評価してください。虚偽や誤解を生む表現を含む応答には低い評価をつけてください。但し、創作や主観的な意見を求める質問の場合、この限りではありません。
流暢性: 応答が自然な文章であるか評価してください。文法的に誤っている応答には低い評価をつけてください。
詳細性: 応答が質問に対して十分な回答を提供しているか評価してください。回答が不足している場合は低い評価をつけてください。
関連性: 応答が質問に関連しているか評価してください。質問と無関係な内容が含まれる場合は低い評価をつけてください。
総合評価: 上記の基準を総合的に評価してください。

評価値は1から5の間です。1は非常に悪く、5は非常に良いことを意味します。
初めに評価の理由を述べ、その後に評価値を記入してください。
評価値を二重角括弧で囲み(例: [[3]])、以下の形式で評価結果を記述してください。

正確性(評価理由): 評価理由を記入
正確性: [[評価値を記入]]

流暢性(評価理由): 評価理由を記入
流暢性: [[評価値を記入]]

詳細性(評価理由): 評価理由を記入
詳細性: [[評価値を記入]]

関連性(評価理由): 評価理由を記入
関連性: [[評価値を記入]]

総合評価(評価理由): 評価理由を記入
総合評価: [[評価値を記入]]

[質問]
{question}

[AIアシスタント回答開始]
{response}
[AIアシスタント回答終了]"""

METRICS = ["正確性", "流暢性", "詳細性", "関連性", "総合評価"]

SCORE_REGEX = f"({'|'.join(METRICS)}):\s?\[\[([1-5])\]\]"


class QualityScoreExtractor(BaseScoreExtractor):
    def __call__(self, text: str) -> dict[str, int]:
        scores = {}
        for metric, score in re.findall(self.regex, text):
            if metric in scores:
                raise ValueError("Duplicate metric")
            scores[metric] = int(score)

        if set(scores.keys()) != set(METRICS):
            raise ValueError("Invalid score format")

        return scores


class QualityEvaluator(BaseEvaluator):
    def log_raw_outputs(self, raw_outputs: list[QualityDatasetItemForEvaluation]):
        if self.dashboard is None:
            return

        table = []
        header = [
            "id",
            "evaluation prompt",
            "evaluation response",
            *METRICS,
            "generate errors",
            "evaluation errors",
        ]
        for raw_output in raw_outputs:
            if raw_output.pattern is None:
                scores = {metric: None for metric in METRICS}
            else:
                scores = [raw_output.pattern[0].get(metric) for metric in METRICS]
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

    def __call__(self, responses: list[QualityDatasetItem]) -> tuple[dict[str, float], dict[str, float]]:
        data = []
        for res in responses:
            d = QualityDatasetItemForEvaluation(
                ID=res.ID,
                prompt=[PROMPT_TEMPLATE.format(question=res.prompt, response=res.response)],
                text=res.text,
                generate_prompt=res.prompt,
                generate_response=res.response,
                generate_errors=res.error_messages,
            )
            data.append(d)

        score_extractor = QualityScoreExtractor(SCORE_REGEX)
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
            for metric, score in raw_output.pattern[0].items():
                scores[metric].append(score)

        self.log_raw_outputs(raw_outputs)

        error_rates = {}
        (
            error_rates[f"{self.name}:api(%)"],
            error_rates[f"{self.name}:pattern_match(%)"],
        ) = self.calc_error_rate(raw_outputs)

        ave_scores = {
            f"quality:{metric}": sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
