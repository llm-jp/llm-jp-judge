import logging


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

    def calc_error_rate(self, raw_outputs):
        api_errors = [raw_output["response"] is None for raw_output in raw_outputs]
        api_error_rate = sum(api_errors) / len(api_errors) * 100

        regex_mismatch_errors = [
            raw_output["pattern"] is None for raw_output in raw_outputs
        ]
        regex_mismatch_rate = (
            sum(regex_mismatch_errors) / len(regex_mismatch_errors) * 100
        )

        logging.info(f"API error rate: {api_error_rate:.2f}%")
        logging.info(f"Pattern mismatch rate: {regex_mismatch_rate:.2f}%")

        self.dashboard.log_summaries(
            {
                f"{self.name}:api error rate(%)": api_error_rate,
                f"{self.name}:regex mismatch rate(%)": regex_mismatch_rate,
            }
        )
