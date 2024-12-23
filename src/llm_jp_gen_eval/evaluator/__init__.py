from .quality import QualityEvaluator
from .safety import SafetyEvaluator
from .mt_bench import MTBenchEvaluator


def load_evaluator(client, dashboard, metric="abs_quality", **kwargs):
    if metric == "abs_quality":
        return QualityEvaluator(client, dashboard, **kwargs)
    elif metric == "abs_safety":
        return SafetyEvaluator(client, dashboard, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
