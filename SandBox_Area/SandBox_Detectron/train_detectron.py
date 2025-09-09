import os
import sys

import sys
import os
sys.path.insert(0, os.path.abspath('detectron2'))

import time
import logging
import datetime
import torch
import numpy as np
from detectron2.engine import DefaultTrainer, HookBase, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds

# Ensure proper CUDA usage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or "1", depending on your setup

# Dataset paths
dataset_name = "dataset"
train_json = "dataset/annotations/train.json"
val_json = "dataset/annotations/val.json"
train_images = "dataset/images/train"
val_images = "dataset/images/val"

# Register datasets
register_coco_instances(f"{dataset_name}_train", {}, train_json, train_images)
register_coco_instances(f"{dataset_name}_val", {}, val_json, val_images)


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._eval_period = eval_period
        self._model = model
        self._data_loader = data_loader

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
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
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
        if self._eval_period > 0 and next_iter % self._eval_period == 0:
            self._do_loss_eval()


class MySegmentationTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True))
        ))
        return hooks_list


# Config setup
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
cfg.DATASETS.TEST = (f"{dataset_name}_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.STEPS = []  # No decay

cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Number of classes in your dataset
cfg.MODEL.MASK_ON = True
cfg.MODEL.DEVICE = "cuda"

cfg.OUTPUT_DIR = "./output_mask_rcnn"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train
trainer = MySegmentationTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
