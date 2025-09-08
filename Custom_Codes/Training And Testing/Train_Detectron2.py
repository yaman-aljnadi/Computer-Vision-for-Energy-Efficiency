
import sys
import os
sys.path.insert(0, os.path.abspath('detectron2'))

from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np
import logging

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, cfg):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.cfg = cfg
        self.best_loss = float("inf")

    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                eta = datetime.timedelta(seconds=int((time.perf_counter() - start_time) / iters_after_start * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f"Loss on Validation done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}",
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            best_model_path = os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth")
            torch.save(self.trainer.model.state_dict(), best_model_path)
            logging.getLogger().info(f"✔ Saved best model to {best_model_path} with val loss: {mean_loss:.4f}")

        return losses

    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return sum(metrics_dict.values())

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class TrainerWithValLoss(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                ),
                self.cfg  
            )
        )
        return hooks_list


register_coco_instances(
    "satellite_train", {}, 
    "Aug_Greyscale_Enhanced_x2/train/_annotations.coco.json", 
    "Aug_Greyscale_Enhanced_x2/train"
)
register_coco_instances(
    "satellite_val", {}, 
    "Aug_Greyscale_Enhanced_x2/valid/_annotations.coco.json", 
    "Aug_Greyscale_Enhanced_x2/valid"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("satellite_train",)
cfg.DATASETS.TEST = ("satellite_val",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 2  
# cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 7000
cfg.TEST.EVAL_PERIOD = 100  
# cfg.SOLVER.STEPS = []       
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  
cfg.INPUT.MIN_SIZE_TRAIN = 2048
cfg.INPUT.MAX_SIZE_TRAIN = 2048
cfg.INPUT.MIN_SIZE_TEST = 2048
cfg.INPUT.MAX_SIZE_TEST = 2048

cfg.OUTPUT_DIR = "./Aug_Greyscale_Enhanced_x2/output_detectron2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = TrainerWithValLoss(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("satellite_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg, trainer.model, evaluators=[evaluator])