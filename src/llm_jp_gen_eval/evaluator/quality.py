import logging

from copy import deepcopy
from collections import defaultdict

SYSTEM_PROMPT = (
    "あなたはAIアシスタントの応答評価において助けになる正確なアシスタントです。"
)
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


class QualityEvaluator:
    def __init__(self, client, dashboard, **kwargs):
        self.client = client
        self.dashboard = dashboard

    def __call__(self, responses):
        data = []
        for res in responses:
            for metric in METRICS:
                d = deepcopy(res)
                d["metric"] = metric
                d["inference_response"] = d["response"]
                d["inference_errors"] = d["error_messages"]
                d["prompt"] = PROMPT_TEMPLATE.format(
                    metric=metric,
                    metric_definition=METRIC_DEFINITIONS[metric],
                    other_metrics=", ".join([m for m in METRICS if m != metric]),
                    question=d["text"],
                    response=d["response"],
                )
                data.append(d)

        raw_scores = self.client(data, regex=SCORE_REGEX, system_prompt=SYSTEM_PROMPT)

        scores = defaultdict(list)
        for raw_score in raw_scores:
            metric = raw_score["metric"]
            if raw_score.get("pattern") is None:
                continue
            score = int(raw_score["pattern"])
            scores[metric].append(score)

        if self.dashboard is not None:
            table = []
            header = [
                "metric",
                "text",
                "inference response",
                "inference errors",
                "score",
                "evaluation response",
                "evaluation errors",
            ]
            for raw_score in raw_scores:
                table.append(
                    [
                        raw_score["metric"],
                        raw_score["text"],
                        raw_score["inference_response"],
                        ", ".join(raw_score["inference_errors"]),
                        raw_score["pattern"],
                        raw_score["response"],
                        ", ".join(raw_score["error_messages"]),
                    ]
                )
            self.dashboard.log("abs_quality_table", columns=header, data=table)

        api_errors = [raw_score["response"] is None for raw_score in raw_scores]
        api_error_rate = sum(api_errors) / len(api_errors * 100)

        pattern_match_errors = [
            raw_score["pattern"] is None for raw_score in raw_scores
        ]
        pattern_match_error_rate = sum(pattern_match_errors) / len(
            pattern_match_errors * 100
        )

        logging.info(f"API error rate: {api_error_rate:.2f}%")
        logging.info(f"Pattern match error rate: {pattern_match_error_rate:.2f}%")

        ave_scores = {
            metric: sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }
        logging.info(f"Scores: {ave_scores}")

        return ave_scores
