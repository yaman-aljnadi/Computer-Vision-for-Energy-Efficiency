import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2

from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.abspath('detectron2'))
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file


setup_logger(output="output", name="detectron2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


DATASET_NAME = "satellite_val"
COCO_JSON = "Aug_Greyscale_Enhanced_x2/test/_annotations.coco.json"
IMAGE_DIR = "Aug_Greyscale_Enhanced_x2/test/"
CHECKPOINT_PATH = "./Aug_Greyscale_Enhanced_x2/output_detectron2/model_final.pth"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
NUM_CLASSES = len(CATEGORY_NAMES)


register_coco_instances(DATASET_NAME, {}, COCO_JSON, IMAGE_DIR)
metadata = MetadataCatalog.get(DATASET_NAME)
dataset_dicts = DatasetCatalog.get(DATASET_NAME)


cfg = get_cfg()
cfg.merge_from_file(get_config_file(CONFIG_FILE))
cfg.DATASETS.TEST = (DATASET_NAME,)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.WEIGHTS = CHECKPOINT_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.INPUT.MIN_SIZE_TEST = 2048
cfg.INPUT.MAX_SIZE_TEST = 2048

predictor = DefaultPredictor(cfg)


def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def conf_matrix_seg(labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    if len(labels) == 0 and len(detections) == 0:
        return confusion_matrix

    
    detections = [d for d in detections if d['score'] > conf_thresh]

    labels_detected = np.zeros(len(labels), dtype=bool)
    detections_matched = np.zeros(len(detections), dtype=bool)

    for l_idx, (gt_mask, gt_class) in enumerate(labels):
        for d_idx, det in enumerate(detections):
            if detections_matched[d_idx]:
                continue
            iou = compute_mask_iou(gt_mask, det['mask'])
            if iou >= iou_thresh:
                confusion_matrix[gt_class, det['class']] += 1
                labels_detected[l_idx] = True
                detections_matched[d_idx] = True
                break

    
    for l_idx, detected in enumerate(labels_detected):
        if not detected:
            confusion_matrix[labels[l_idx][1], -1] += 1

    
    for d_idx, matched in enumerate(detections_matched):
        if not matched:
            confusion_matrix[-1, detections[d_idx]['class']] += 1

    return confusion_matrix


confusion_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    height, width = img.shape[:2]
    outputs = predictor(img)

    
    labels = []
    for ann in d["annotations"]:
        rles = mask_utils.frPyObjects(ann["segmentation"], height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle).astype(bool)
        labels.append((mask, ann["category_id"]))

    
    pred_masks = outputs["instances"].pred_masks.cpu().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    detections = []
    for mask, cls, score in zip(pred_masks, pred_classes, scores):
        detections.append({'mask': mask.astype(bool), 'class': cls, 'score': score})

    
    confusion_matrix += conf_matrix_seg(labels, detections, NUM_CLASSES, conf_thresh=0.25, iou_thresh=0.7)


class_names = CATEGORY_NAMES + ["Background"]
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
print(df_cm)
