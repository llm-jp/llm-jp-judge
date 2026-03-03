from src.llm_jp_judge.client.base import BaseClient
from src.llm_jp_judge.dashboard.base import BaseDashboard
from src.llm_jp_judge.evaluator.mt_bench import MTBenchEvaluator
from src.llm_jp_judge.evaluator.quality import QualityEvaluator
from src.llm_jp_judge.evaluator.safety import SafetyEvaluator


def load_evaluator(
    client: BaseClient,
    dashboard: BaseDashboard,
    metadata: dict[str, str] | None = None,
    metric: str = "abs_quality",
    **kwargs,
):
    if metadata is None:
        metadata = {}

    if metric == "quality":
        return QualityEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety":
        return SafetyEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
