from torch.optim import lr_scheduler

from . import config


def get(cfg, optimizer):
    """Get annealing object from annealing config.

    Args:
      cfg: Config, just for annealing.
      optimizer: pytorch optimizer object.

    Returns:
      pytorch annealing object if annealing, or None if cfg.sched is None.
    """
    if not cfg.schedule:
        return None
    elif cfg.schedule == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=cfg.factor,
            patience=cfg.patience)
    else:
        raise ValueError(f'Unexpected lr_schedule: {cfg.schedule}')


class AnnealingConfig(config.Config):

    def __init__(self, schedule, epoch, iter):
        """Create a new AnnealingConfig.

        Args:
          schedule: String, defines the schedule.
          epoch: Bool, whether to execute each epoch.
          iter: Bool, whether to execute each iteration.
        """
        super().__init__()
        self.schedule = schedule
        self.epoch = epoch
        self.iter = iter


class NoAnnealing(config.Config):

    def __init__(self):
        super().__init__(schedule=None, epoch=False, iter=False)


class ReduceLROnPlateauConfig(config.Config):

    def __init__(self, factor, patience):
        super().__init__(schedule='plateau', epoch=True, iter=False)
        self.factor = factor
        self.patience = patience
