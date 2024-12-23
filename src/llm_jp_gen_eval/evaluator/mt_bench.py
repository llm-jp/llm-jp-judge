import logging

from copy import deepcopy
from collections import defaultdict

from ..utils.data import load_jsonl


class MTBenchEvaluator:
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
        query["generate_errors"] = query["error_messages"]
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
                kwargs["ref_answer_1"] = self.references[response["ID"]][0]
                kwargs["ref_answer_2"] = self.references[response["ID"]][1]
            else:
                query["metric"] = "single-v1-multi-turn"
        else:
            query["turn"] = 1
            kwargs = {
                "question": response["prompt"][0],
                "answer": response["response"][0],
            }
            if use_reference:
                query["metric"] = "single-math-v1"
                kwargs["ref_answer_1"] = self.references[response["ID"]][0]
            else:
                query["metric"] = "single-v1"

        prompt_template = self.prompt_template[query["metric"]]["prompt_template"]
        query["prompt"] = prompt_template.format(**kwargs)
        return query

    def calc_score(self, raw_scores):
        raw_scores = [r for r in raw_scores if r.get("pattern") is not None]
        t1_raw_scores = [r for r in raw_scores if r["turn"] == 1]
        t2_raw_scores = [r for r in raw_scores if r["turn"] == 2]

        # Evaluate average score
        ave_score = sum([int(r["pattern"]) for r in raw_scores]) / len(raw_scores)
        logging.info(f"Average score: {ave_score:.2f}")

        # Evaluate turn-wise scores
        header = ["generate model", "evaluation model", "turn 1", "turn 2", "average"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]

        t1_score = sum([int(r["pattern"]) for r in t1_raw_scores]) / len(t1_raw_scores)
        t2_score = sum([int(r["pattern"]) for r in t2_raw_scores]) / len(t2_raw_scores)
        
        logging.info(f"Average score (turn 1): {t1_score:.2f}")
        logging.info(f"Average score (turn 2): {t2_score:.2f}")

        row.append(t1_score)
        row.append(t2_score)
        row.append(ave_score)
        self.dashboard.log(f"{self.name}_turn_score_table", columns=header, data=[row])
        
        # Evaluate category-wise scores
        categ_raw_scores = defaultdict(list)
        for raw_score in raw_scores:
            categ_raw_scores[raw_score["category"]].append(int(raw_score["pattern"]))

        header = ["generate model", "evaluation model"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]
        for categ in sorted(categ_raw_scores.keys()):
            categ_score = sum(categ_raw_scores[categ]) / len(categ_raw_scores[categ])
            header.append(categ)
            row.append(categ_score)
            logging.info(f"Average score (caetegory {categ}): {categ_score:.2f}")

        header.append("average")
        row.append(ave_score)
        self.dashboard.log(f"{self.name}_category_score_table", columns=header, data=[row])

        return ave_score

    def evaluate(self, responses, use_reference=False, multi_turn=False):
        if len(responses) == 0:
            return []

        queries = []
        for response in responses:
            query = self.conv_to_query(response, use_reference, multi_turn)
            queries.append(query)

        metric = queries[-1]["metric"]
        responses = self.client(
            queries,
            system_prompt=self.prompt_template[metric]["system_prompt"],
            sampling_params=self.sampling_params,
            regex=self.prompt_template[metric]["regex"],
        )
        return responses

    def __call__(self, responses):
        questions_ref = [
            r for r in responses if r["category"] in self.reference_categories
        ]
        questions = [
            r for r in responses if r["category"] not in self.reference_categories
        ]

        raw_scores = []
        # Single-turn evaluation
        raw_scores += self.evaluate(questions, use_reference=False, multi_turn=False)
        raw_scores += self.evaluate(questions_ref, use_reference=True, multi_turn=False)

        # Multi-turn evaluation
        raw_scores += self.evaluate(questions, use_reference=False, multi_turn=True)
        raw_scores += self.evaluate(questions_ref, use_reference=True, multi_turn=True)

        api_errors = [raw_score["response"] is None for raw_score in raw_scores]
        api_error_rate = sum(api_errors) / len(api_errors) * 100

        pattern_match_errors = [
            raw_score["pattern"] is None for raw_score in raw_scores
        ]
        pattern_match_error_rate = (
            sum(pattern_match_errors) / len(pattern_match_errors) * 100
        )

        logging.info(f"API error rate: {api_error_rate:.2f}%")
        logging.info(f"Pattern match error rate: {pattern_match_error_rate:.2f}%")

        ave_scores = {}
        ave_scores[self.name] = self.calc_score(raw_scores)

        return ave_scores
