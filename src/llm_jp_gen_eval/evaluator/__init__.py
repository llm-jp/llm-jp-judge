from .quality import QualityEvaluator
from .safety import SafetyEvaluator


def load_evaluator(client, dashboard, name="quality", **kwargs):
    if name == "quality":
        return QualityEvaluator(client, dashboard, **kwargs)
    elif name == "safety":
        return SafetyEvaluator(client, dashboard, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {name}")
