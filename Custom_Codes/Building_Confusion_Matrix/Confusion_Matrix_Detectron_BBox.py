import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.abspath('detectron2'))
print(torch.cuda.memory_summary())

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, pairwise_iou
from detectron2.model_zoo import get_config_file


setup_logger(output="output", name="detectron2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


DATASET_NAME = "satellite_val"
COCO_JSON = "Aug_Original/test/_annotations.coco.json"
IMAGE_DIR = "Aug_Original/test/"
CHECKPOINT_PATH = "./Aug_Original/output_detectron2/model_final.pth"
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
cfg.INPUT.MIN_SIZE_TEST = 1280
cfg.INPUT.MAX_SIZE_TEST = 1280

predictor = DefaultPredictor(cfg)


evaluator = COCOEvaluator(DATASET_NAME, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, DATASET_NAME)
inference_on_dataset(predictor.model, val_loader, evaluator)


def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out

def conf_matrix_calc(labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
    if len(labels) == 0 and len(detections) == 0:
        return confusion_matrix

    if len(labels) > 0:
        l_classes = np.array(labels)[:, 0].astype(int)
        l_bboxs = coco_bbox_to_coordinates(np.array(labels)[:, 1:])
    else:
        l_classes = np.array([])
        l_bboxs = np.empty((0, 4))

    if len(detections) > 0:
        d_confs = np.array(detections)[:, 4]
        detections = np.array(detections)[d_confs > conf_thresh]
        d_bboxs = detections[:, :4]
        d_classes = detections[:, -1].astype(int)
    else:
        d_bboxs = np.empty((0, 4))
        d_classes = np.array([])

    labels_detected = np.zeros(len(labels))
    detections_matched = np.zeros(len(detections))

    for l_idx, (l_class, l_bbox) in enumerate(zip(l_classes, l_bboxs)):
        for d_idx, (d_bbox, d_class) in enumerate(zip(d_bboxs, d_classes)):
            iou = pairwise_iou(
                Boxes(torch.from_numpy(np.array([l_bbox]))),
                Boxes(torch.from_numpy(np.array([d_bbox])))
            )[0][0].item()
            if iou >= iou_thresh and not detections_matched[d_idx]:
                confusion_matrix[l_class, d_class] += 1
                labels_detected[l_idx] = 1
                detections_matched[d_idx] = 1
                break

    for i in np.where(labels_detected == 0)[0]:
        confusion_matrix[l_classes[i], -1] += 1  
    for i in np.where(detections_matched == 0)[0]:
        confusion_matrix[-1, d_classes[i]] += 1  

    return confusion_matrix


confusion_matrix = np.zeros([NUM_CLASSES + 1, NUM_CLASSES + 1])
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    labels = [[ann["category_id"]] + ann["bbox"] for ann in d["annotations"]]
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    detections = [list(box) + [score] + [cls] for box, score, cls in zip(pred_boxes, scores, classes)]
    confusion_matrix += conf_matrix_calc(labels, detections, NUM_CLASSES, conf_thresh=0.25, iou_thresh=0.7)


class_names = CATEGORY_NAMES + ["Background"]
df_cm = pd.DataFrame(confusion_matrix.astype(int), index=class_names, columns=class_names)
print(df_cm)
