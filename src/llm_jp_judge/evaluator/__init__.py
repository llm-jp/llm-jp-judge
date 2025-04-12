from .quality import QualityEvaluator
from .safety import SafetyEvaluator
from .borderline import BorderlineEvaluator
from .mt_bench import MTBenchEvaluator


def load_evaluator(client, dashboard, metadata={}, metric="abs_quality", **kwargs):
    if metric == "quality":
        return QualityEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety":
        return SafetyEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "borderline":
        return BorderlineEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
