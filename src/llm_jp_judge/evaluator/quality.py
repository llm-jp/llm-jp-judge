import re
import json
import logging

from copy import deepcopy
from collections import defaultdict

from .base import BaseEvaluator, ScoreExtractionError

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


class QualityScoreExtractor(object):
    def __init__(self, regex):
        self.regex = regex

    def __call__(self, text):
        scores = {}
        for metric, score in re.findall(self.regex, text):
            scores[metric] = score

        if set(scores.keys()) != set(METRICS):
            raise ScoreExtractionError("Invalid score format")

        return scores


class QualityEvaluator(BaseEvaluator):
    def __init__(self, *args, empty_response_score=None, **kwargs):
        self.empty_response_score = empty_response_score
        super().__init__(*args, **kwargs)

    def log_raw_outputs(self, raw_outputs):
        if self.dashboard is None:
            return

        table = []
        header = [
            "id",
            "generation prompt",
            "generation response",
            "evaluation prompt",
            "evaluation response",
            *[f"pattern:{metric}" for metric in METRICS],
            *[f"score:{metric}" for metric in METRICS],
            "generation errors",
            "evaluation errors",
        ]
        for raw_output in raw_outputs:
            if raw_output.get("pattern") is None:
                patterns = [None] * len(METRICS)
            else:
                patterns = [raw_output.get("pattern", {}).get(metric) for metric in METRICS]
            scores = [raw_output.get("score", {}).get(metric) for metric in METRICS]
            table.append(
                [
                    raw_output["ID"],
                    raw_output["generation_prompt"],
                    raw_output["generation_response"],
                    raw_output["prompt"],
                    raw_output["response"],
                    *patterns,
                    *scores,
                    json.dumps(raw_output["generation_errors"], ensure_ascii=False),
                    json.dumps(raw_output["error_messages"], ensure_ascii=False),
                ]
            )
        self.dashboard.log_table("quality_raw_output_table", columns=header, data=table)

    def __call__(self, responses):
        data = []
        skip_outputs = []
        for res in responses:
            d = deepcopy(res)
            d["generation_prompt"] = d["prompt"]
            d["generation_response"] = d["response"]
            d["generation_errors"] = d.get("error_messages", [])
            d["prompt"] = PROMPT_TEMPLATE.format(
                question=d["prompt"],
                response=d["response"],
            )

            if d["response"] is None or d["response"].strip() == "":
                if self.empty_response_score is not None:
                    # 評価対象の応答が空の場合は、empty_response_score(デフォルトは1)とする。
                    d["score"] = {metric: int(self.empty_response_score) for metric in METRICS}
                    skip_outputs.append(d)
                    continue
            data.append(d)

        score_extractor = QualityScoreExtractor(SCORE_REGEX)
        raw_outputs = self.client(
            data,
            score_extractor=score_extractor,
            system_prompt=self.system_prompt,
            sampling_params=self.sampling_params,
        )

        error_rates = {}
        (
            error_rates[f"{self.name}:api(%)"],
            error_rates[f"{self.name}:pattern_match(%)"],
        ) = self.calc_error_rate(raw_outputs)

        # 最終スコアの計算
        for raw_output in raw_outputs:
            raw_output["score"] = {}
            for metric in METRICS:
                if raw_output.get("pattern") is None or raw_output.get("pattern", {}).get(metric) is None:
                    raw_output["score"][metric] = None
                    continue
                raw_output["score"][metric] = int(raw_output["pattern"][metric])

        raw_outputs += skip_outputs
        self.log_raw_outputs(raw_outputs)

        # スコアの集計
        scores = defaultdict(list)
        for raw_output in raw_outputs:
            for metric in METRICS:
                if raw_output["score"][metric] is not None:
                    scores[metric].append(raw_output["score"][metric])

        ave_scores = {
            f"quality:{metric}": sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
