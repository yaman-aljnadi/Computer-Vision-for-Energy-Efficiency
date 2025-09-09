def main():
    import os
    import sys
    import torch
    import numpy as np
    import pandas as pd
    import cv2

    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    sys.path.insert(0, os.path.abspath('../Adnan_Tasks/detectron2'))
    print(torch.cuda.memory_summary())

    # Detectron2 imports
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

    # COCO API
    from pycocotools.coco import COCO

    # Setup
    setup_logger(output="output", name="detectron2")

    # CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Paths and Config
    DATASET_NAME = "satellite_val"
    VAL_JSON = "../Models_Training/Detectron2/Aug_Greyscale_Enhanced_x2/test/_annotations.coco.json"
    VAL_IMAGES = "../Models_Training/Detectron2/Aug_Greyscale_Enhanced_x2/test/"
    CHECKPOINT_PATH = "../Logs/Seg_Logs/Detectron2/Detectron2_Seg_aug_Greyscale_Enhanced_2x/model_final.pth"
    CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    SAVE_DIR = "Aug_Greyscale_Enhanced_x2/predicted_images"
    CATEGORY_NAMES = ['None','Garage', 'House', 'Other', 'Propane', 'Trailer']
    PROPANE_CLASS_ID = 4


    os.makedirs(SAVE_DIR, exist_ok=True)

    # Register dataset if not yet registered
    val_name = f"{DATASET_NAME}_val"

    if val_name not in DatasetCatalog.list():
        register_coco_instances(val_name, {}, VAL_JSON, VAL_IMAGES)

    # Load COCO annotations
    coco = COCO(VAL_JSON)

    # Metadata
    metadata = MetadataCatalog.get(val_name)

    # Config
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

    # Predictor
    predictor = DefaultPredictor(cfg)

    # Evaluator
    evaluator = COCOEvaluator(val_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, val_name)
    eval_results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="output")

    # Iterate through the dataset
    results = []
    image_stats = []
    
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
        keep = (instances_pred.pred_classes == PROPANE_CLASS_ID)
        instances_pred_propane = instances_pred[keep]

        # Only visualize if there is at least one Propane prediction
        if len(instances_pred_propane) > 0:
            v_pred = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
            vis_pred = v_pred.draw_instance_predictions(instances_pred_propane)
        else:
            vis_pred = None
            print(f"No Propane predictions found in {file_name}")

        # Load ground-truth annotations
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        masks = []
        boxes_gt = []
        classes_gt = []

        for ann in anns:
            category_id = ann["category_id"]

            if category_id == PROPANE_CLASS_ID:
                mask = coco.annToMask(ann)
                masks.append(mask)

                bbox = ann["bbox"]
                x, y, w, h = bbox
                boxes_gt.append([x, y, x + w, y + h])

                classes_gt.append(category_id)

        num_pred_propane = len(instances_pred_propane)
        num_gt_propane = len(boxes_gt)

        image_stats.append({
            "file_name": os.path.basename(file_name),
            "num_predicted_propane": num_pred_propane,
            "num_gt_propane": num_gt_propane,
        })

        # Proceed only if we have at least one Propane GT object
        if len(boxes_gt) > 0:
            mask_tensor = torch.stack([torch.from_numpy(m.astype(np.uint8)) for m in masks]) \
                if len(masks) > 0 else torch.empty((0, image_np.shape[0], image_np.shape[1]), dtype=torch.uint8)

            bitmasks = BitMasks(mask_tensor)
            boxes_tensor = torch.tensor(boxes_gt, dtype=torch.float32)

            instances_gt = Instances(image_np.shape[:2])
            instances_gt.gt_boxes = Boxes(boxes_tensor)
            instances_gt.gt_classes = torch.tensor(classes_gt, dtype=torch.int64)
            instances_gt.gt_masks = bitmasks

            v_gt = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)

            target_fields = instances_gt.get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]

            vis_gt = v_gt.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
            )
        else:
            vis_gt = None
            print(f"No Propane ground-truth annotations found for image {file_name}")

        # Combine prediction and ground truth if both are available
        if vis_pred is not None and vis_gt is not None:
            pred_img_rgb = vis_pred.get_image()
            gt_img_rgb = vis_gt.get_image()

            pred_img_bgr = cv2.cvtColor(pred_img_rgb, cv2.COLOR_RGB2BGR)
            gt_img_bgr = cv2.cvtColor(gt_img_rgb, cv2.COLOR_RGB2BGR)

            # Ensure same height
            if pred_img_bgr.shape[0] != gt_img_bgr.shape[0]:
                height = min(pred_img_bgr.shape[0], gt_img_bgr.shape[0])
                pred_img_bgr = cv2.resize(pred_img_bgr, (pred_img_bgr.shape[1], height))
                gt_img_bgr = cv2.resize(gt_img_bgr, (gt_img_bgr.shape[1], height))

            combined_image = cv2.hconcat([pred_img_bgr, gt_img_bgr])

            out_path_combined = os.path.join(SAVE_DIR, f"COMBINED_{os.path.basename(file_name)}")
            cv2.imwrite(out_path_combined, combined_image)
        elif vis_pred is not None:
            pred_img_rgb = vis_pred.get_image()
            pred_img_bgr = cv2.cvtColor(pred_img_rgb, cv2.COLOR_RGB2BGR)
            out_path_pred = os.path.join(SAVE_DIR, f"PRED_{os.path.basename(file_name)}")
            cv2.imwrite(out_path_pred, pred_img_bgr)
        elif vis_gt is not None:
            gt_img_rgb = vis_gt.get_image()
            gt_img_bgr = cv2.cvtColor(gt_img_rgb, cv2.COLOR_RGB2BGR)
            out_path_gt = os.path.join(SAVE_DIR, f"GT_{os.path.basename(file_name)}")
            cv2.imwrite(out_path_gt, gt_img_bgr)


    if len(image_stats) > 0:
        df_stats = pd.DataFrame(image_stats)
        df_stats.to_csv("Aug_Greyscale_Enhanced_x2/Propane_instance_counts.csv", index=False)
        print("Per-image propane counts saved to CSV.")
    else:
        print("No propane instances to record.")

if __name__ == "__main__":
    main()