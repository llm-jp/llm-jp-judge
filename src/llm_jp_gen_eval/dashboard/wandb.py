import wandb

import omegaconf


class WandB(object):
    def __init__(self, cfg, entity=None, project=None, run_name=None):
        assert entity is not None, "dashboard.entity is required for dashboard=wandb"
        assert project is not None, "dashboard.project is required for dashboard=wandb"

        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        self.run = wandb.init(project=project, entity=entity, name=run_name)

    def close(self):
        self.run.finish()

    def log(self, name, columns=[], data=[]):
        table = wandb.Table(columns=columns, data=data)
        self.run.log({name: table})
