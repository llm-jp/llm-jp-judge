import logging


class BaseEvaluator:
    def calc_error_rate(self, raw_scores):
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
