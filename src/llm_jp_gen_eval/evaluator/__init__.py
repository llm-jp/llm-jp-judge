from .quality import QualityEvaluator
from .safety import SafetyEvaluator


def load_evaluator(client, name="quality", **kwargs):
    if name == "quality":
        return QualityEvaluator(client, **kwargs)
    elif name == "safety":
        return SafetyEvaluator(client, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {name}")
