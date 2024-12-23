import logging

from copy import deepcopy

from ..utils.data import load_jsonl


class MTBenchEvaluator:
    def __init__(
        self,
        client,
        dashboard,
        name="mt_bench",
        mode="single",
        prompt_template=None,
        sampling_params={},
        reference={"path": None, "categories": None},
        **kwargs,
    ):
        self.client = client
        self.dashboard = dashboard
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

    def calc_score(self, responses):
        responses = [r for r in responses if r.get("pattern") is not None]
        t1_responses = [r for r in responses if r["turn"] == 1]
        t2_responses = [r for r in responses if r["turn"] == 2]

        ave_t1_score = sum([int(r["pattern"]) for r in t1_responses]) / len(
            t1_responses
        )
        ave_t2_score = sum([int(r["pattern"]) for r in t2_responses]) / len(
            t2_responses
        )
        ave_score = sum([int(r["pattern"]) for r in responses]) / len(responses)

        logging.info(f"Average score (turn 1): {ave_t1_score:.2f}")
        logging.info(f"Average score (turn 2): {ave_t2_score:.2f}")
        logging.info(f"Average score: {ave_score:.2f}")

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

        responses = []
        # Single-turn evaluation
        responses += self.evaluate(questions, use_reference=False, multi_turn=False)
        responses += self.evaluate(questions_ref, use_reference=True, multi_turn=False)

        # Multi-turn evaluation
        responses += self.evaluate(questions, use_reference=False, multi_turn=True)
        responses += self.evaluate(questions_ref, use_reference=True, multi_turn=True)

        ave_scores = {}
        ave_scores[self.name] = self.calc_score(responses)

        return ave_scores
