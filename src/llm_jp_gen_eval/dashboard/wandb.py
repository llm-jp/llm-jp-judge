import wandb

import omegaconf

from .base import BaseDashboard


class WandB(BaseDashboard):
    def __init__(self, cfg, entity=None, project=None, run_name=None):
        super().__init__()

        assert entity is not None, "dashboard.entity is required for dashboard=wandb"
        assert project is not None, "dashboard.project is required for dashboard=wandb"

        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        self.run = wandb.init(project=project, entity=entity, name=run_name)

    def close(self):
        self.run.finish()

    def log(self, data):
        super().log(data)
        self.run.log(data)

    def log_table(self, name, columns=[], data=[]):
        table = wandb.Table(columns=columns, data=data)
        self.log({name: table})

    def log_summary(self, key, value):
        super().log_summary(key, value)
        self.run.summary[key] = value

    def log_summaries(self, data):
        for key, value in data.items():
            self.log_summary(key, value)
