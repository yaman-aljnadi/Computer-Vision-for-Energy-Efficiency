import os
import numpy as np
import cv2
import mmcv
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from mmdet.structures.bbox import bbox_overlaps
import pandas as pd
import torch

config_file = '/home/data/files/Yaman/Satalite_Images/Models_Training/MMDetection/Aug_Greyscale/mask_rcnn_custom.py'
checkpoint_file = '/home/data/files/Yaman/Satalite_Images/Models_Training/MMDetection/Aug_Greyscale/mask_rcnn_custom_V2/epoch_20.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')


coco_gt = COCO('Aug_Greyscale/test/_annotations.coco.json')
image_ids = coco_gt.getImgIds()
class_names = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
num_classes = len(class_names)

def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def conf_matrix_calc_mm(gt_annos, pred_boxes, pred_labels, pred_scores, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1))  

    gt_classes = np.array([ann['category_id'] for ann in gt_annos])
    gt_bboxes = np.array([coco_bbox_to_xyxy(ann['bbox']) for ann in gt_annos])
    
    if len(gt_bboxes) == 0:
        gt_bboxes = np.empty((0, 4))
    if len(pred_boxes) == 0:
        pred_boxes = np.empty((0, 4))
        pred_labels = np.array([])
        pred_scores = np.array([])

    valid_preds = pred_scores >= conf_thresh
    pred_boxes = pred_boxes[valid_preds]
    pred_labels = pred_labels[valid_preds]

    matched_gt = np.zeros(len(gt_bboxes), dtype=bool)
    matched_pred = np.zeros(len(pred_boxes), dtype=bool)

    if len(gt_bboxes) > 0 and len(pred_boxes) > 0:
        ious = bbox_overlaps(torch.tensor(pred_boxes, dtype=torch.float32), torch.tensor(gt_bboxes, dtype=torch.float32))  

        for i in range(len(pred_boxes)):
            max_iou = 0
            max_j = -1
            for j in range(len(gt_bboxes)):
                if matched_gt[j]:
                    continue
                iou = ious[i, j]
                if iou > max_iou and iou >= iou_thresh:
                    max_iou = iou
                    max_j = j
            if max_j >= 0:
                gt_label = gt_classes[max_j]
                pred_label = pred_labels[i]
                confusion_matrix[gt_label, pred_label] += 1
                matched_gt[max_j] = True
                matched_pred[i] = True

    
    for i, matched in enumerate(matched_gt):
        if not matched:
            confusion_matrix[gt_classes[i], -1] += 1  

    
    for i, matched in enumerate(matched_pred):
        if not matched:
            confusion_matrix[-1, pred_labels[i]] += 1  

    return confusion_matrix

conf_matrix = np.zeros((num_classes + 1, num_classes + 1))  
for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join('Aug_Greyscale/test', img_info['file_name'])
    gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    gt_annos = coco_gt.loadAnns(gt_ann_ids)

    result = inference_detector(model, img_path)

    
    pred_instances = result.pred_instances  
    if pred_instances is not None and len(pred_instances.bboxes) > 0:
        pred_boxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()
    else:
        pred_boxes = np.empty((0, 4))
        pred_scores = np.array([])
        pred_labels = np.array([])

    conf_matrix += conf_matrix_calc_mm(
        gt_annos, pred_boxes, pred_labels, pred_scores,
        num_classes, conf_thresh=0.25, iou_thresh=0.7
    )

class_names_with_bg = class_names + ['Background']
df = pd.DataFrame(conf_matrix.astype(int), index=class_names_with_bg, columns=class_names_with_bg)
print(df)