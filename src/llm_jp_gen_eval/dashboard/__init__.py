from .wandb import WandB


def load_dashboard(cfg, name=None, **kwargs):
    if name is None:
        return NullDashboard()
    elif name == "wandb":
        return WandB(cfg, **kwargs)
    else:
        raise ValueError(f"Invalid dashboard name: {name}")


def NullDashboard():
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass

        return method
