from typing import Any

from llm_jp_judge.client.base import BaseClient
from llm_jp_judge.dashboard.base import BaseDashboard
from llm_jp_judge.evaluator.mt_bench import MTBenchEvaluator
from llm_jp_judge.evaluator.quality import QualityEvaluator
from llm_jp_judge.evaluator.safety import SafetyEvaluator


def load_evaluator(
    client: BaseClient, dashboard: BaseDashboard, metadata: dict[str, Any] = {}, metric: str = "abs_quality", **kwargs
):
    if metric == "quality":
        return QualityEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety":
        return SafetyEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
