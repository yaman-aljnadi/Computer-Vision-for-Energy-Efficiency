import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.abspath('detectron2'))
print(torch.cuda.memory_summary())
from torch.utils.data import DataLoaderz
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.model_zoo import get_config_file
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, pairwise_iou

from torch.utils.tensorboard import SummaryWriter



DATASET_NAME = "satellite_val"
COCO_JSON = "Aug_Greyscale_Original/test/_annotations.coco.json"
IMAGE_DIR = "Aug_Greyscale_Original/test/"
CHECKPOINT_PATH = "./Aug_Greyscale_Original/output_detectron2/model_final.pth"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
RESULT_CSV = "Aug_Greyscale_Original/MaskRCNN_segmentation_results.csv"
CATEGORY_NAMES = ['None','Garage', 'House', 'Other', 'Propane', 'Trailer']

setup_logger(output="output", name="detectron2")

datasets = ['satellite_val']

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CHECKPOINT_DIR = "Aug_Greyscale_Original/MaskRCNN_segmentation_results"

CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

Results = {}

writer = SummaryWriter(log_dir="output")

for dataset in datasets:
    print(f"Testing on dataset: {dataset}")

    val_json = f"Aug_Greyscale_Original/test/_annotations.coco.json"
    val_images = f"Aug_Greyscale_Original/test/"

    val_name = f"{dataset}_val"
    
    if val_name not in DatasetCatalog.list():
        register_coco_instances(val_name, {}, val_json, val_images)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG_FILE))
    cfg.DATASETS.TEST = (val_name,)
    cfg.MODEL.WEIGHTS = CHECKPOINT_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 1280
    cfg.INPUT.MAX_SIZE_TEST = 1280

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(val_name, cfg, False, output_dir="./output/")

    val_loader = build_detection_test_loader(cfg, val_name)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    if "bbox" in eval_results:
        metrics = eval_results["bbox"]
        ap50 = metrics.get("AP50", 0)
        ap75 = metrics.get("AP75", 0)
        ap = metrics.get("AP", 0)

        try:
            precision = evaluator._results["bbox"]["precision"]  
            recall = evaluator._results["bbox"]["recall"]  
        except KeyError:
            precision = None
            recall = None

        if precision is not None and recall is not None:
            final_precision = precision[-1].mean()  
            final_recall = recall[-1].mean()  

            TP = int(final_recall * metrics["gt"])  
            FP = int((1 - final_precision) * TP)  
            FN = int(metrics["gt"] - TP) 

            BBoxes = int(metrics.get("dt", 0))  
        else:
            TP, FP, FN, final_precision, final_recall, BBoxes = [None] * 6

        Results[dataset] = {
            "AP50": ap50,
            "AP75": ap75,
            "AP": ap,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Pr": final_precision,
            "Rc": final_recall,
            "BBoxes": BBoxes
        }

        print(f"Evaluation results for {dataset}: {Results[dataset]}")

        writer.add_scalar(f"{dataset}/AP50", ap50)
        writer.add_scalar(f"{dataset}/AP75", ap75)
        writer.add_scalar(f"{dataset}/AP", ap)
        if TP is not None:
            writer.add_scalar(f"{dataset}/TP", TP)
            writer.add_scalar(f"{dataset}/FP", FP)
            writer.add_scalar(f"{dataset}/FN", FN)
            writer.add_scalar(f"{dataset}/Precision", final_precision)
            writer.add_scalar(f"{dataset}/Recall", final_recall)
            writer.add_scalar(f"{dataset}/BBoxes", BBoxes)


df = pd.DataFrame.from_dict(Results, orient="index")

df.to_csv("Aug_Greyscale_Original/Mask_RCNN_detection_results.csv")
print(df)

writer.close()
