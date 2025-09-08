
import os
import torch
import pandas as pd
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('detectron2'))
print(torch.cuda.memory_summary())
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
from pycocotools.coco import COCO

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.model_zoo import get_config_file
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, BoxMode, Instances, BitMasks

DATASET_NAME = "satellite_val"
COCO_JSON = "../Models_Training/Detectron2/Aug_Greyscale_Enhanced_x2/test/_annotations.coco.json"
IMAGE_DIR = "../Models_Training/Detectron2/Aug_Greyscale_Enhanced_x2/test/"
CHECKPOINT_PATH = "./Aug_Greyscale_Enhanced_x2/output_detectron2/model_final.pth"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
RESULT_CSV = "Aug_Greyscale_Enhanced_x2/MaskRCNN_segmentation_results.csv"
CATEGORY_NAMES = ['None','Garage', 'House', 'Other', 'Propane', 'Trailer']

setup_logger(output="output", name="detectron2")

datasets = ['satellite_val']

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CHECKPOINT_DIR = "Aug_Greyscale_Enhanced_x2/MaskRCNN_segmentation_results"

CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

Results = {}

writer = SummaryWriter(log_dir="output")

for dataset in datasets:
    print(f"Testing on dataset: {dataset}")

    val_json = f"Aug_Greyscale_Enhanced_x2/test/_annotations.coco.json"
    val_images = f"Aug_Greyscale_Enhanced_x2/test/"

    val_name = f"{dataset}_val"
    
    if val_name not in DatasetCatalog.list():
        register_coco_instances(val_name, {}, val_json, val_images)

    coco = COCO(COCO_JSON)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG_FILE))
    cfg.DATASETS.TEST = (val_name,)
    cfg.MODEL.WEIGHTS = CHECKPOINT_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048
    

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(val_name, cfg, False, output_dir="./output/")

    val_loader = build_detection_test_loader(cfg, val_name)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)


    save_dir = "Aug_Greyscale_Enhanced_x2/predicted_images"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    metadata = MetadataCatalog.get(val_name)

for i, data in enumerate(build_detection_test_loader(cfg, val_name)):
    # Data for one image
    image_data = data[0]
    file_name = image_data["file_name"]
    image_id = image_data["image_id"]

    # Read image
    image_np = cv2.imread(file_name)

    # Run prediction
    outputs = predictor(image_np)
    instances_pred = outputs["instances"].to("cpu")

    # Save prediction visualization
    v_pred = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
    vis_pred = v_pred.draw_instance_predictions(instances_pred)
    out_path_pred = os.path.join(CHECKPOINT_DIR, f"PRED_{os.path.basename(file_name)}")
    cv2.imwrite(out_path_pred, vis_pred.get_image()[:, :, ::-1])

    # Save prediction results to CSV (optional)
    boxes = instances_pred.pred_boxes.tensor.numpy()
    scores = instances_pred.scores.numpy()
    classes = instances_pred.pred_classes.numpy()

    for box, score, cls in zip(boxes, scores, classes):
        results.append({
            "file": os.path.basename(file_name),
            "class_id": int(cls),
            "class_name": CATEGORY_NAMES[int(cls)],
            "score": float(score),
            "bbox_xmin": float(box[0]),
            "bbox_ymin": float(box[1]),
            "bbox_xmax": float(box[2]),
            "bbox_ymax": float(box[3]),
        })

    # Load ground-truth annotations
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    masks = []
    boxes_gt = []
    classes_gt = []

    for ann in anns:
        mask = coco.annToMask(ann)
        masks.append(mask)

        bbox = ann["bbox"]
        x, y, w, h = bbox
        boxes_gt.append([x, y, x + w, y + h])

        category_id = ann["category_id"]
        classes_gt.append(category_id)

    if len(anns) > 0:
        # Convert to Detectron2 BitMasks
        bitmasks = BitMasks(torch.stack([torch.from_numpy(m).bool() for m in masks]))
        boxes_tensor = torch.tensor(boxes_gt, dtype=torch.float32)

        instances_gt = Instances(image_np.shape[:2])
        instances_gt.pred_boxes = Boxes(boxes_tensor)
        instances_gt.pred_classes = torch.tensor(classes_gt, dtype=torch.int64)
        instances_gt.pred_masks = bitmasks

        # Visualize ground-truth
        v_gt = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_gt = v_gt.draw_instance_predictions(instances_gt)
        out_path_gt = os.path.join(CHECKPOINT_DIR, f"GT_{os.path.basename(file_name)}")
        cv2.imwrite(out_path_gt, vis_gt.get_image()[:, :, ::-1])
    else:
        print(f"No annotations found for image {file_name}")

# Save all results to CSV
if len(results) > 0:
    df = pd.DataFrame(results)
    df.to_csv("Aug_Greyscale_Enhanced_x2/MaskRCNN_segmentation_results.csv", index=False)
    print("Results saved to CSV.")
else:
    print("No predictions to save.")
