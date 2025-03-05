from .wandb import WandB
from .base import BaseDashboard


def load_dashboard(cfg, name=None, **kwargs):
    if name is None:
        return BaseDashboard()
    elif name == "wandb":
        return WandB(cfg, **kwargs)
    else:
        raise ValueError(f"Invalid dashboard name: {name}")
