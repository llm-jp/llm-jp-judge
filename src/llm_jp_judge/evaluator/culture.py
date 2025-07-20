import logging

from copy import deepcopy
from collections import defaultdict

from .base import BaseEvaluator, BaseScoreExtractor

PROMPT_TEMPLATE = """[指示]
公平かつ客観的な判断者として行動し、以下に示すユーザの質問に対するAIアシスタントの回答の正確性を評価してください。
AIアシスタントの回答に加え、模範的な回答である参考回答が与えられます。参考回答と比較したうえで、AIアシスタントの回答の正確性を評価してください。
評価は短い説明から始めてください。説明を提供した後、「評価：[[評価値]]」という形式で1から5までの尺度で応答を評価してください（例：評価：[[5]]）。
各尺度の基準は以下の通りです。
5: 正確かつ非常に有用
4: 正確だが、正確性以外の点に軽微な問題あり
3: 正確だが、有用ではないまたは正確性以外の点に重大な問題あり
2: 軽度または中度の正確性の問題あり
1: 重大な正確性の問題あり
ただし正確性の評価については非常に厳しく行う必要があり、たとえ回答の大部分が正確かつ有用であっても、一部に正確性の問題があれば1または2を選択してください。

[質問]
{question}

[参考回答開始]
{reference}
[参考回答終了]

[AIアシスタント回答開始]
{response}
[AIアシスタント回答終了]"""

SCORE_REGEX = r"\[\[(\d+)\]\]"


class CultureEvaluator(BaseEvaluator):
    def __init__(
        self, *args, api_error_score=None, empty_response_score=None, **kwargs
    ):
        self.api_error_score = api_error_score
        self.empty_response_score = empty_response_score
        super().__init__(*args, **kwargs)

    def __call__(self, responses):
        data = []
        skipped_outputs = []
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

            if d["generate_response"] is None or d["generate_response"] == "":
                if self.empty_response_score is not None:
                    # 評価対象の応答が空の場合は、評価値はempty_response_score(デフォルトは1)とする。
                    d["score"] = int(self.empty_response_score)
                    skipped_outputs.append(d)
                    continue

            data.append(d)

        score_extractor = BaseScoreExtractor(regex=SCORE_REGEX)
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
        for raw_score in raw_outputs:
            if raw_score.get("pattern") is None:
                raw_score["score"] = None
                continue
            raw_score["score"] = int(raw_score["pattern"])

        raw_outputs += skipped_outputs
        self.log_raw_outputs(raw_outputs)

        scores = defaultdict(list)
        for raw_output in raw_outputs:
            metric = raw_output["metric"]

            if raw_output.get("score") is None:
                continue
            scores[metric].append(raw_output["score"])

        ave_scores = {
            f"culture:{metric}": sum(scores) / len(scores) if len(scores) else None
            for metric, scores in scores.items()
        }

        is_incorrect = [score < 3 for score in scores["日本文化"]]
        ave_scores["culture:違反回答率(%)"] = (
            sum(is_incorrect) / len(is_incorrect) * 100
        )

        is_useful = [score > 3 for score in scores["日本文化"]]
        ave_scores["culture:許容回答率(%)"] = sum(is_useful) / len(is_useful) * 100

        logging.info(f"Scores: {ave_scores}")

        return ave_scores, error_rates
