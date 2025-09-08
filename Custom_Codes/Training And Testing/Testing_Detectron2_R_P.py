
import os
import torch
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('detectron2'))
print(torch.cuda.memory_summary())
from torch.utils.data import DataLoader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.model_zoo import get_config_file
from detectron2.utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval
import numpy as np
import logging

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode
from detectron2.utils.logger import setup_logger
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import torch
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    # box1, box2: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def build_confusion_matrix(predictor, dataset_name, iou_thresh=0.5):
    metadata = MetadataCatalog.get(dataset_name)
    thing_classes = metadata.thing_classes
    num_classes = len(thing_classes)

    gt_dataset = DatasetCatalog.get(dataset_name)

    y_true = []
    y_pred = []

    for data in tqdm(gt_dataset, desc="Evaluating"):
        image = cv2.imread(data["file_name"])
        outputs = predictor(image)
        pred_instances = outputs["instances"].to("cpu")

        pred_boxes = pred_instances.pred_boxes.tensor.numpy()
        pred_classes = pred_instances.pred_classes.numpy()

        gt_anns = data["annotations"]
        gt_boxes = [BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in gt_anns]
        gt_classes = [obj["category_id"] for obj in gt_anns]

        matched_gt = set()

        for pb, pc in zip(pred_boxes, pred_classes):
            best_iou = 0
            best_gt_idx = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_thresh:
                gt_cls = gt_classes[best_gt_idx]
                y_true.append(gt_cls)
                y_pred.append(pc)
                matched_gt.add(best_gt_idx)
            else:
                # False positive, no matching GT
                y_true.append(num_classes)  
                y_pred.append(pc)

        # False negatives (missed ground truth)
        for i, gt_cls in enumerate(gt_classes):
            if i not in matched_gt:
                y_true.append(gt_cls)
                y_pred.append(num_classes)  

    labels = list(range(num_classes)) + [num_classes]
    label_names = thing_classes + ["background"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion Matrix @ IoU {iou_thresh}")
    plt.tight_layout()
    plt.show()

    return cm

DATASET_NAME = "satellite_val"
COCO_JSON = "Aug_Original/test/_annotations.coco.json"
IMAGE_DIR = "Aug_Original/test/"
CHECKPOINT_PATH = "./Aug_Original/output_detectron2/model_final.pth"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
RESULT_CSV = "Aug_Original/MaskRCNN_segmentation_results.csv"
CATEGORY_NAMES = ['None','Garage', 'House', 'Other', 'Propane', 'Trailer']

setup_logger(output="output", name="detectron2")

datasets = ['satellite_val']

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CHECKPOINT_DIR = "Aug_Original/MaskRCNN_segmentation_results"

CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

Results = {}

writer = SummaryWriter(log_dir="output")

for dataset in datasets:
    print(f"Testing on dataset: {dataset}")

    val_json = f"Aug_Original/test/_annotations.coco.json"
    val_images = f"Aug_Original/test/"

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

    cm = build_confusion_matrix(predictor, "satellite_val", iou_thresh=0.5)

    # val_loader = build_detection_test_loader(cfg, val_name)
    # eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

