from omegaconf import DictConfig

from llm_jp_judge.dashboard.base import BaseDashboard
from llm_jp_judge.dashboard.wandb import WandB


def load_dashboard(cfg: DictConfig, name: str = None, **kwargs) -> BaseDashboard:
    if name is None:
        return BaseDashboard()
    elif name == "wandb":
        return WandB(cfg, **kwargs)
    else:
        raise ValueError(f"Invalid dashboard name: {name}")
