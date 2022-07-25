import wandb

from utils import dist_utils
from .base import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, logdir, run_id=None, resume=False):
        self.is_master = dist_utils.is_master_process()
        if not self.is_master:
            return

        self.logdir = logdir
        if not resume:
            if run_id is None:
                run_id = wandb.util.generate_id()
            resume = run_id
        self.experiment = wandb.init(
            dir=self.logdir,
            resume=resume,
        )
        self.run_id = self.experiment.id
        self.experiment.define_metric("train/step")
        self.experiment.define_metric("*", step_metric="train/step")
    
    def log_metrics(self, metrics, step=None):
        if not self.is_master:
            return
        if step is not None:
            self.experiment.log({**metrics, "train/step": step})
        else:
            self.experiment.log(metrics)
    
    def state_dict(self):
        if not self.is_master:
            return
        state_dict = {
            "run_id": self.run_id,
            "logdir": self.logdir,
        }
        return state_dict