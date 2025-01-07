import logging

from copy import deepcopy
from collections import defaultdict

from .base import BaseEvaluator

PROMPT_TEMPLATE = """[指示]
質問に対するAIアシスタントの回答を以下の指標に基づいて評価してください。
{metric} (1-5) : {metric_definition}
この際、その他の指標({other_metrics})は考慮しないでください。
評価値は1から5の間です。1は非常に悪く、5は非常に良いことを意味します。
初めに評価の理由を述べ、その後に評価値を記入してください。
評価は評価: [[評価値]]のように記入してください。
評価は慎重に行い、専門家の評価に近いものにする必要があります。

[質問]
{question}

[AIアシスタント回答開始]
{response}
[AIアシスタント回答終了]"""

SCORE_REGEX = r"\[\[(\d)\]\]"

METRIC_DEFINITIONS = {
    "流暢性": "応答が自然な文章であるか評価してください。文法的に誤っている応答には低い評価をつけてください。",
    "正確性": "応答が事実を述べているか評価してください。虚偽や誤解を生む表現を含む応答には低い評価をつけてください。但し、創作や主観的な意見を求める質問の場合、この限りではありません。",
    "詳細性": "応答が質問に対して十分な回答を提供しているか評価してください。回答が不足している場合は低い評価をつけてください。",
    "関連性": "応答が質問に関連しているか評価してください。質問と無関係な内容が含まれる場合は低い評価をつけてください。",
}

METRICS = sorted(METRIC_DEFINITIONS.keys())


class QualityEvaluator(BaseEvaluator):
    def __call__(self, responses):
        data = []
        for res in responses:
            for metric in METRICS:
                d = deepcopy(res)
                d["metric"] = metric
                d["generate_response"] = d["response"]
                d["generate_errors"] = d.get("error_messages", [])
                d["prompt"] = PROMPT_TEMPLATE.format(
                    metric=metric,
                    metric_definition=METRIC_DEFINITIONS[metric],
                    other_metrics=", ".join([m for m in METRICS if m != metric]),
                    question=d["text"],
                    response=d["response"],
                )
                data.append(d)

        raw_outputs = self.client(
            data,
            regex=SCORE_REGEX,
            system_prompt=self.system_prompt,
            sampling_params=self.sampling_params,
        )

        scores = defaultdict(list)
        for raw_output in raw_outputs:
            metric = raw_output["metric"]
            if raw_output.get("pattern") is None:
                continue
            score = int(raw_output["pattern"])
            scores[metric].append(score)

        if self.dashboard is not None:
            table = []
            header = [
                "id",
                "metric",
                "text",
                "generate response",
                "generate errors",
                "score",
                "evaluation response",
                "evaluation errors",
            ]
            for raw_output in raw_outputs:
                table.append(
                    [
                        raw_output["ID"],
                        raw_output["metric"],
                        raw_output["text"],
                        raw_output["generate_response"],
                        ", ".join(raw_output["generate_errors"]),
                        raw_output["pattern"],
                        raw_output["response"],
                        ", ".join(raw_output["error_messages"]),
                    ]
                )
            self.dashboard.log_table("abs_quality_table", columns=header, data=table)

        self.calc_error_rate(raw_outputs)

        ave_scores = {
            metric: sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores
