import json
import logging
from collections import defaultdict
from copy import deepcopy

from ..utils.data import load_jsonl
from .base import BaseEvaluator, BaseScoreExtractor


class MTBenchEvaluator(BaseEvaluator):
    def __init__(
        self,
        client,
        dashboard,
        metadata={},
        name="mt_bench",
        mode="single",
        prompt_template=None,
        sampling_params={},
        reference={"path": None, "categories": None},
        **kwargs,
    ):
        self.client = client
        self.dashboard = dashboard
        self.metadata = metadata
        self.name = name

        if mode not in ["single"]:
            raise ValueError("Invalid mode for MTBenchEvaluator: {mode}")
        self.mode = mode

        if prompt_template is None:
            raise ValueError("prompt_template is required for MTBenchEvaluator")
        self.prompt_template = prompt_template

        self.references = None
        if reference["path"] is not None:
            data = load_jsonl(reference["path"])
            self.references = {d["question_id"]: d["choices"][0]["turns"] for d in data}

        self.reference_categories = reference["categories"]
        self.sampling_params = sampling_params

    def conv_to_query(self, response, use_reference=False, multi_turn=False):
        query = deepcopy(response)
        query["generate_response"] = query["response"]
        query["generate_errors"] = query.get("error_messages", [])
        if multi_turn:
            query["turn"] = 2
            kwargs = {
                "question_1": response["prompt"][0],
                "question_2": response["prompt"][1],
                "answer_1": response["response"][0],
                "answer_2": response["response"][1],
            }
            if use_reference:
                query["metric"] = "single-math-v1-multi-turn"
                query["use_reference"] = True
                kwargs["ref_answer_1"] = self.references[response["ID"]][0]
                kwargs["ref_answer_2"] = self.references[response["ID"]][1]
            else:
                query["use_reference"] = False
                query["metric"] = "single-v1-multi-turn"
        else:
            query["turn"] = 1
            kwargs = {
                "question": response["prompt"][0],
                "answer": response["response"][0],
            }
            if use_reference:
                query["use_reference"] = True
                query["metric"] = "single-math-v1"
                kwargs["ref_answer_1"] = self.references[response["ID"]][0]
            else:
                query["use_reference"] = False
                query["metric"] = "single-v1"

        prompt_template = self.prompt_template[query["metric"]]["prompt_template"]
        query["prompt"] = prompt_template.format(**kwargs)
        query["system_prompt"] = self.prompt_template[query["metric"]]["system_prompt"]
        return query

    def calc_score(self, raw_outputs):
        raw_outputs = [r for r in raw_outputs if r.get("pattern") is not None]

        # Evaluate average score
        ave_score = sum([int(r["pattern"]) for r in raw_outputs]) / len(raw_outputs)
        logging.info(f"Average score: {ave_score:.2f}")

        # Evaluate turn-wise scores
        t1_raw_outputs = [r for r in raw_outputs if r["turn"] == 1]
        t2_raw_outputs = [r for r in raw_outputs if r["turn"] == 2]

        t1_score = sum([int(r["pattern"]) for r in t1_raw_outputs]) / len(
            t1_raw_outputs
        )
        t2_score = sum([int(r["pattern"]) for r in t2_raw_outputs]) / len(
            t2_raw_outputs
        )

        logging.info(f"Average score (turn 1): {t1_score:.2f}")
        logging.info(f"Average score (turn 2): {t2_score:.2f}")

        header = ["generation_model", "evaluation_model", "turn 1", "turn 2", "average"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]

        row.append(t1_score)
        row.append(t2_score)
        row.append(ave_score)
        self.dashboard.log_table(
            f"{self.name}_turn_score_table", columns=header, data=[row]
        )

        # Evaluate category-wise scores
        categ_raw_outputs = defaultdict(list)
        for raw_output in raw_outputs:
            categ_raw_outputs[raw_output["category"]].append(int(raw_output["pattern"]))

        header = ["generation_model", "evaluation_model"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]
        for categ in sorted(categ_raw_outputs.keys()):
            categ_score = sum(categ_raw_outputs[categ]) / len(categ_raw_outputs[categ])
            header.append(categ)
            row.append(categ_score)
            logging.info(f"Average score (category {categ}): {categ_score:.2f}")

        header.append("average")
        row.append(ave_score)
        self.dashboard.log_table(
            f"{self.name}_category_score_table", columns=header, data=[row]
        )

        return ave_score

    def log_raw_outputs(self, raw_outputs):
        if self.dashboard is None:
            return

        columns = [
            "id",
            "category",
            "metric",
            "turn",
            "use reference",
            "system prompt",
            "prompt",
            "response",
            "score",
            "generate errors",
            "evaluation errors",
        ]
        data = [
            [
                score["ID"],
                score["category"],
                score["metric"],
                score["turn"],
                score["use_reference"],
                score["system_prompt"],
                score["prompt"],
                score["response"],
                score["pattern"],
                json.dumps(score["generate_errors"], ensure_ascii=False),
                json.dumps(score["error_messages"], ensure_ascii=False),
            ]
            for score in raw_outputs
        ]
        return self.dashboard.log_table(
            f"{self.name}_raw_output_table", columns=columns, data=data
        )

    def evaluate(self, responses, use_reference=False, multi_turn=False):
        if len(responses) == 0:
            return []

        queries = []
        for response in responses:
            query = self.conv_to_query(response, use_reference, multi_turn)
            queries.append(query)

        metric = queries[-1]["metric"]
        score_extractor = BaseScoreExtractor(
            regex=self.prompt_template[metric]["regex"]
        )
        responses = self.client(
            queries,
            score_extractor=score_extractor,
            system_prompt=self.prompt_template[metric]["system_prompt"],
            sampling_params=self.sampling_params,
        )
        return responses

    def __call__(self, responses):
        questions_ref = [
            r for r in responses if r["category"] in self.reference_categories
        ]
        questions = [
            r for r in responses if r["category"] not in self.reference_categories
        ]

        raw_outputs = []
        # Single-turn evaluation
        raw_outputs += self.evaluate(questions, use_reference=False, multi_turn=False)
        raw_outputs += self.evaluate(
            questions_ref, use_reference=True, multi_turn=False
        )

        # Multi-turn evaluation
        raw_outputs += self.evaluate(questions, use_reference=False, multi_turn=True)
        raw_outputs += self.evaluate(questions_ref, use_reference=True, multi_turn=True)

        self.log_raw_outputs(raw_outputs)

        error_rates = {}
        (
            error_rates[f"{self.name}:api(%)"],
            error_rates[f"{self.name}:pattern_match(%)"],
        ) = self.calc_error_rate(raw_outputs)

        ave_scores = {}
        ave_scores[self.name] = self.calc_score(raw_outputs)

        return ave_scores, error_rates
