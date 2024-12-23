from .quality import QualityEvaluator
from .safety import SafetyEvaluator
from .mt_bench import MTBenchEvaluator


def load_evaluator(client, dashboard, metadata={}, metric="abs_quality", **kwargs):
    if metric == "abs_quality":
        return QualityEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "abs_safety":
        return SafetyEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
