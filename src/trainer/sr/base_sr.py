import torch
import torch.nn.functional as F

from trainer.base import BaseTrainer
from utils import dist_utils
from utils.metrics import Evaluator
from utils.common import ld2dl


class SRTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = Evaluator(
            metrics=["psnr"], 
            data_range=2.0
        )
    
    def calc_loss(self, img_hr, img_sr):
        loss = F.l1_loss(img_hr, img_sr)
        return loss
    
    def training_step(self, data):
        img_hr, img_lr = data["img_hr"], data["img_lr"]
        img_lr = img_lr.to(self.device)
        img_hr = img_hr.to(self.device)
        img_sr = self.model(img_lr)
        loss = self.calc_loss(img_hr, img_sr)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        metrics = {"loss": loss}
        description = f"Loss: {loss.item():.4f}"
        return metrics, description
    
    @torch.no_grad()
    def validation_step(self, data):
        img_hr, img_lr = data["img_hr"], data["img_lr"]
        img_lr = img_lr.to(self.device)
        img_hr = img_hr.to(self.device)
        img_sr = self.model(img_lr)

        img_hr = dist_utils.gather_and_concat(img_hr)
        img_sr = dist_utils.gather_and_concat(img_sr)

        val_result = self.evaluator(img_hr, img_sr)
        return val_result
    
    def validation_epoch_end(self, outputs):
        outputs = ld2dl(outputs)
        for k in outputs:
            outputs[k] = torch.mean(torch.stack(outputs[k]))
        return outputs
        