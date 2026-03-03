import json
import logging
from collections import defaultdict

from omegaconf import DictConfig

from src.llm_jp_judge.client.base import BaseClient
from src.llm_jp_judge.dashboard.base import BaseDashboard
from src.llm_jp_judge.dataset.mt_bench import MTBenchDatasetItem, MTBenchDatasetItemForEvaluation
from src.llm_jp_judge.evaluator.base import BaseEvaluator, BaseScoreExtractor
from src.llm_jp_judge.utils.data import load_jsonl


class MTBenchEvaluator(BaseEvaluator):
    def __init__(
        self,
        client: BaseClient,
        dashboard: BaseDashboard,
        metadata: dict[str, str] | None = None,
        name: str = "mt_bench",
        mode: str = "single",
        prompt_template: dict[str, dict[str, str]] | DictConfig | None = None,
        sampling_params: dict[str, dict[int | float | None]] | DictConfig | None = None,
        reference: dict[str, str | list[str]] | DictConfig | None = None,
        **kwargs,
    ):
        if metadata is None:
            metadata = {}
        if sampling_params is None:
            sampling_params = {}
        if reference is None:
            reference = {"path": None, "categories": None}

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

    def conv_to_query(
        self, response: MTBenchDatasetItem, use_reference: bool = False, multi_turn: bool = False
    ) -> MTBenchDatasetItemForEvaluation:
        if multi_turn:
            turn = 2
            kwargs = {
                "question_1": response.prompt[0],
                "question_2": response.prompt[1],
                "answer_1": response.response[0],
                "answer_2": response.response[1],
            }
            if use_reference:
                metric = "single-math-v1-multi-turn"
                kwargs["ref_answer_1"] = self.references[response.ID][0]
                kwargs["ref_answer_2"] = self.references[response.ID][1]
            else:
                metric = "single-v1-multi-turn"
        else:
            turn = 1
            kwargs = {
                "question": response.prompt[0],
                "answer": response.response[0],
            }
            if use_reference:
                metric = "single-math-v1"
                kwargs["ref_answer_1"] = self.references[response.ID][0]
            else:
                metric = "single-v1"

        prompt_template = self.prompt_template[metric]["prompt_template"]
        prompt = prompt_template.format(**kwargs)
        system_prompt = self.prompt_template[metric]["system_prompt"]

        query = MTBenchDatasetItemForEvaluation(
            ID=response.ID,
            prompt=[prompt],
            category=response.category,
            generate_response=response.response,
            generate_errors=response.error_messages,
            metric=metric,
            turn=turn,
            use_reference=use_reference,
            system_prompt=system_prompt,
        )

        return query

    def calc_score(self, raw_outputs: list[MTBenchDatasetItem]) -> float:
        raw_outputs = [r for r in raw_outputs if r.pattern[0] is not None]

        # Evaluate average score
        ave_score = sum([int(r.pattern[0]) for r in raw_outputs]) / len(raw_outputs)
        logging.info(f"Average score: {ave_score:.2f}")

        # Evaluate turn-wise scores
        t1_raw_outputs = [r for r in raw_outputs if r.turn == 1]
        t2_raw_outputs = [r for r in raw_outputs if r.turn == 2]

        t1_score = sum([int(r.pattern[0]) for r in t1_raw_outputs]) / len(t1_raw_outputs)
        t2_score = sum([int(r.pattern[0]) for r in t2_raw_outputs]) / len(t2_raw_outputs)

        logging.info(f"Average score (turn 1): {t1_score:.2f}")
        logging.info(f"Average score (turn 2): {t2_score:.2f}")

        header = ["generation_model", "evaluation_model", "turn 1", "turn 2", "average"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]

        row.append(t1_score)
        row.append(t2_score)
        row.append(ave_score)
        self.dashboard.log_table(f"{self.name}_turn_score_table", columns=header, data=[row])

        # Evaluate category-wise scores
        categ_raw_outputs = defaultdict(list)
        for raw_output in raw_outputs:
            categ_raw_outputs[raw_output.category].append(int(raw_output.pattern[0]))

        header = ["generation_model", "evaluation_model"]
        row = [self.metadata.get("model_name", "N/A"), self.client.model_name]
        for categ in sorted(categ_raw_outputs.keys()):
            categ_score = sum(categ_raw_outputs[categ]) / len(categ_raw_outputs[categ])
            header.append(categ)
            row.append(categ_score)
            logging.info(f"Average score (category {categ}): {categ_score:.2f}")

        header.append("average")
        row.append(ave_score)
        self.dashboard.log_table(f"{self.name}_category_score_table", columns=header, data=[row])

        return ave_score

    def log_raw_outputs(self, raw_outputs: list[MTBenchDatasetItemForEvaluation]):
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
                score.ID,
                score.category,
                score.metric,
                score.turn,
                score.use_reference,
                score.system_prompt,
                score.prompt[0],
                score.response[0],
                score.pattern[0],
                json.dumps(score.generate_errors, ensure_ascii=False),
                json.dumps(score.error_messages[0], ensure_ascii=False),
            ]
            for score in raw_outputs
        ]
        self.dashboard.log_table(f"{self.name}_raw_output_table", columns=columns, data=data)

    def evaluate(
        self,
        responses: list[MTBenchDatasetItem],
        use_reference: bool = False,
        multi_turn: bool = False,
    ) -> list[MTBenchDatasetItemForEvaluation]:
        if len(responses) == 0:
            return []

        queries = []
        for response in responses:
            query = self.conv_to_query(response, use_reference, multi_turn)
            queries.append(query)

        metric = queries[-1].metric
        score_extractor = BaseScoreExtractor(regex=self.prompt_template[metric]["regex"])
        responses = self.client(
            queries,
            score_extractor=score_extractor,
            system_prompt=self.prompt_template[metric]["system_prompt"],
            sampling_params=self.sampling_params,
        )
        return responses

    def __call__(self, responses: list[MTBenchDatasetItem]) -> tuple[dict[str, float], dict[str, float]]:
        questions_ref = [r for r in responses if r.category in self.reference_categories]
        questions = [r for r in responses if r.category not in self.reference_categories]

        raw_outputs = []
        # Single-turn evaluation
        raw_outputs += self.evaluate(questions, use_reference=False, multi_turn=False)
        raw_outputs += self.evaluate(questions_ref, use_reference=True, multi_turn=False)

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
