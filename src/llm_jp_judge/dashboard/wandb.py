from typing import Any, cast

import omegaconf
import wandb
from omegaconf import DictConfig

from .base import BaseDashboard


class WandB(BaseDashboard):
    def __init__(
        self, cfg: DictConfig, entity: str | None = None, project: str | None = None, run_name: str | None = None
    ):
        super().__init__()

        assert entity is not None, "dashboard.entity is required for dashboard=wandb"
        assert project is not None, "dashboard.project is required for dashboard=wandb"

        wandb.config = cast(
            wandb.sdk.wandb_config.Config, omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        self.run = wandb.init(project=project, entity=entity, name=run_name)

    def close(self):
        self.run.finish()

    def log(self, data: dict[str, Any]):
        super().log(data)

        self.run.log(data)

    def log_table(self, name: str, columns: list[str] | None = None, data: list[list[Any]] | None = None):
        if columns is None:
            columns = []
        if data is None:
            data = []

        super().log_table(name, columns, data)

        table = wandb.Table(columns=columns, data=data)
        self.run.log({name: table})

    def log_summary(self, key: str, value: Any):
        super().log_summary(key, value)

        self.run.summary[key] = value

    def log_summaries(self, data: dict[str, Any]):
        super().log_summaries(data)

        for key, value in data.items():
            self.run.summary[key] = value
