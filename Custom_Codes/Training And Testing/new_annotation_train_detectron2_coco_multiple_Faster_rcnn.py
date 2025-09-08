
import sys
import os
sys.path.insert(0, os.path.abspath('detectron2'))

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds

from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, DatasetMapper
import detectron2.utils.comm as comm
import logging
import torch
import time
import datetime
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#datasets = ["ACDC", "AutoEnhancer", "BayesianRetinex", "ICSP", "Original", "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]
datasets = ["ACDC", "BayesianRetinex", "ICSP", "PCDE", "Semi_UIR", "TEBCF", "TUDA", "USUIR"]

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
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
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,  # Add this argument
                    "Loss on Validation done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_metric = 0.0  # Store the best metric (AP)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.append(BestModelSaverHook(self))
        return hooks_list


class BestModelSaverHook(hooks.HookBase):
    def __init__(self, trainer):
        self.trainer = trainer
        self.best_metric = 0.0  # Track the best metric

    def after_step(self):
	#This function doesn't work, just don't bother with it
        # Evaluate only at the set interval
        if self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            evaluator = COCOEvaluator(self.trainer.cfg.DATASETS.TEST[0], self.trainer.cfg, False)
            val_loader = build_detection_test_loader(self.trainer.cfg, self.trainer.cfg.DATASETS.TEST[0])
            results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

            # Get the validation metric (bbox/AP in COCO format)
            metric = results["bbox"]["AP"] if "bbox" in results else 0.0

            # Save the best model if AP improves
            if metric > self.best_metric:
                self.best_metric = metric
                print(f"New best model found with AP: {metric:.4f}! Saving...")
                checkpointer = DetectionCheckpointer(self.trainer.model, save_dir=self.trainer.cfg.OUTPUT_DIR)
                checkpointer.save("model_best")  # Saves as model_best.pth

for dataset in datasets:
    print(f"Training on dataset: {dataset}")

    train_json = f"Enhanced_RUOD_coco/{dataset}/new_annotations/new_train_fixed.json"
    val_json = f"Enhanced_RUOD_coco/{dataset}/new_annotations/new_val_fixed.json"
    train_images = f"Enhanced_RUOD_coco/{dataset}/images/train"
    val_images = f"Enhanced_RUOD_coco/{dataset}/images/val"

    train_name = f"{dataset}_train"
    val_name = f"{dataset}_val"
    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name, {}, val_json, val_images)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 16
    # cfg.SOLVER.MAX_ITER = 150000
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.TEST.EVAL_PERIOD = 200
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 2.0
    # cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"

    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.STEPS = (30000,) 
    cfg.SOLVER.GAMMA = 0.1  

    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  

    cfg.OUTPUT_DIR = f"new_annotations_output_faster_rcnn/{dataset}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
