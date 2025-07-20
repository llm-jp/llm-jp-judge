import re
import json
import logging

class ScoreExtractionError(Exception):
    pass


class BaseScoreExtractor(object):
    def __init__(self, regex):
        self.regex = regex

    def __call__(self, text):
        m = re.search(self.regex, text)
        if m is None:
            raise ScoreExtractionError(f"Regex '{self.regex}' did not match.")
        return m.group(1)


class BaseEvaluator:
    def __init__(
        self,
        client,
        dashboard,
        metadata={},
        name="base",
        use_reference=False,
        system_prompt=None,
        sampling_params={},
    ):
        self.client = client
        self.dashboard = dashboard
        self.name = name
        self.metadata = metadata
        self.use_reference = use_reference
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params

    def log_raw_outputs(self, raw_outputs):
        if self.dashboard is None:
            return

        columns = [
            "id",
            "metric",
            "generation prompt",
            "generation response",
            "evaluation prompt",
            "evaluation response",
            "pattern",
            "score",            
            "generation errors",
            "evaluation errors",
        ]
        data = [
            [
                score["ID"],
                score["metric"],
                score["generation_prompt"],
                score["generation_response"],
                score["prompt"],
                score["response"],
                score["pattern"],
                score["score"],
                json.dumps(score["generation_errors"]),
                json.dumps(score["error_messages"]),
            ]
            for score in raw_outputs
        ]
        return self.dashboard.log_table(
            f"{self.name}_raw_output_table", columns=columns, data=data
        )

    def calc_error_rate(self, raw_outputs):
        api_errors = [raw_output["response"] is None for raw_output in raw_outputs]
        api_error_rate = sum(api_errors) / len(api_errors) * 100

        regex_match_errors = [
            raw_output["pattern"] is None for raw_output in raw_outputs
        ]
        regex_match_error_rate = sum(regex_match_errors) / len(regex_match_errors) * 100

        logging.info(f"API error rate: {api_error_rate:.2f}%")
        logging.info(f"Pattern match error rate: {regex_match_error_rate:.2f}%")

        return api_error_rate, regex_match_error_rate
