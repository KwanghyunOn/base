import os
from abc import ABCMeta, abstractmethod

import torch
from tqdm import tqdm

from utils import dist_utils


class BaseTrainer(metaclass=ABCMeta):
    def __init__(
        self,
        model,
        optimizer,
        loader_train,
        loader_val,
        logger,
        root,
        max_epochs,
        log_every_n_steps,
        val_every_n_epochs,
        save_ckpt_every_n_epochs,
        distributed=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.logger = logger

        self.root = root
        self.ckpt_path = os.path.join(self.root, "checkpoints")
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.distributed = distributed
        self.is_master = dist_utils.is_master_process()

        self.global_step = 0
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_epochs = val_every_n_epochs
        self.save_ckpt_every_n_epochs = save_ckpt_every_n_epochs

        log_dir = os.path.join(self.root, "logs")
        os.makedirs(log_dir, exist_ok=True)

    @abstractmethod
    def training_step(self, data):
        pass

    @abstractmethod
    def validation_step(self, data):
        pass

    @abstractmethod
    def validation_epoch_end(self, outputs):
        pass

    def _get_state_dict(self):
        if self.distributed:
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def _load_state_dict(self, state_dict):
        if self.distributed:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def _save_checkpoint(self):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model": self._get_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "logger": self.logger.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.ckpt_path, "latest.ckpt"))

    def _load_checkpoint(self):
        ckpt = torch.load(
            os.path.join(self.ckpt_path, "latest.ckpt"),
            map_location="cpu",
        )
        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self._load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def run(self, resume=False):
        if resume:
            self._load_checkpoint()

        while self.current_epoch < self.max_epochs:
            if self.distributed:
                self.loader_train.sampler.set_epoch(self.current_epoch)

            self.model.train()
            torch.set_grad_enabled(True)
            loader_train_tq = tqdm(
                self.loader_train,
                disable=(not self.is_master),
                ncols=100,
                mininterval=0.1,
            )
            for data in loader_train_tq:
                metrics, description = self.training_step(data)
                self.global_step += 1
                if self.global_step % self.log_every_n_steps == 0:
                    self.logger.log_metrics(metrics, step=self.global_step)
                loader_train_tq.set_description(
                    f"[Epoch: {self.current_epoch}, {description}]",
                    refresh=False,  # respect mininterval
                )
            dist_utils.wait()

            self.model.eval()
            torch.set_grad_enabled(False)
            val_outputs = []
            for data in self.loader_val:
                val_outputs.append(self.validation_step(data))
            if self.current_epoch % self.val_every_n_epochs == 0:
                val_output = self.validation_epoch_end(val_outputs)
                self.logger.log_metrics(val_output, step=self.global_step)

            if (
                self.is_master
                and self.current_epoch % self.save_ckpt_every_n_epochs == 0
            ):
                self._save_checkpoint()
            self.current_epoch += 1
            dist_utils.wait()
