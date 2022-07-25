import os
import shutil
import yaml


class Config:
    def __init__(self, cfg_path, resume, reset):
        self.cfg_path = cfg_path
        self.resume = resume
        self.reset = reset
        with open(self.cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self._set_dirs()

    def _set_dirs(self):
        self.root = os.path.join(
            self.__call__("exp", "root"),
            self.__call__("exp", "name"),
            self.__call__("exp", "ablation"),
        )
        ckpt_path = os.path.join(self.root, "ckpts", "latest.ckpt")
        if self.reset:
            shutil.rmtree(self.root, ignore_errors=True)
        elif os.path.exists(ckpt_path) and not self.resume:
            raise Exception(
                "Checkpoint found. Add --resume to resume the experiment or --reset to remove the existing training results."
            )
        os.makedirs(self.root, exist_ok=True)
        self.logdir = os.path.join(self.root, "logs")
        os.makedirs(self.logdir, exist_ok=True)
        shutil.copyfile(self.cfg_path, os.path.join(self.root, "config.yaml"))

    def __call__(self, *keys, default=None):
        d = self.cfg
        for key in list(keys):
            if key not in d:
                return default
            else:
                d = d[key]
        return d
