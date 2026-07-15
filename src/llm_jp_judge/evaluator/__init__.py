from ..client.base import BaseClient
from ..dashboard.base import BaseDashboard
from .base import BaseEvaluator
from .culture import CultureEvaluator
from .mt_bench import MTBenchEvaluator
from .quality import QualityEvaluator
from .safety import SafetyEvaluator
from .safety_borderline import SafetyBorderlineEvaluator
from .safety_boundary import SafetyBoundaryEvaluator


def load_evaluator(
    client: BaseClient,
    dashboard: BaseDashboard,
    metadata: dict[str, str] | None = None,
    metric: str = "abs_quality",
    **kwargs,
) -> BaseEvaluator:
    if metadata is None:
        metadata = {}

    if metric == "quality":
        return QualityEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety":
        return SafetyEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "culture":
        return CultureEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety_borderline":
        return SafetyBorderlineEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "safety_boundary":
        return SafetyBoundaryEvaluator(client, dashboard, metadata=metadata, **kwargs)
    elif metric == "mt_bench":
        return MTBenchEvaluator(client, dashboard, metadata=metadata, **kwargs)
    else:
        raise ValueError(f"Invalid evaluator name: {metric}")
