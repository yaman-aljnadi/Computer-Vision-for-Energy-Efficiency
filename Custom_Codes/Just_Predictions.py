def main():
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
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import cv2
    from glob import glob

    DATASET_NAME = "satellite_val"
    IMAGE_DIR = "Images/2048"  
    COCO_JSON = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Detectron2_Seg_aug_Original/test/_annotations.coco.json"
    CHECKPOINT_PATH = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Logs/Seg_Logs/Detectron2/Detectron2_Seg_aug_Original/model_final.pth"
    CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    SAVE_DIR = "Aug_Enhanced_2x/predicted_background/2048"
    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']

    setup_logger(output="output", name="detectron2")
    os.makedirs(SAVE_DIR, exist_ok=True)


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png"))

    if DATASET_NAME not in DatasetCatalog.list():
            register_coco_instances(DATASET_NAME, {}, COCO_JSON, IMAGE_DIR)

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG_FILE))
    cfg.MODEL.WEIGHTS = CHECKPOINT_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048

    predictor = DefaultPredictor(cfg)


    metadata = MetadataCatalog.get(DATASET_NAME) 
    metadata.set(thing_classes=CATEGORY_NAMES)
    results = []


    for file_path in image_paths:
        print(f"Processing: {file_path}")
        image_np = cv2.imread(file_path)

        outputs = predictor(image_np)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        for box, score, cls in zip(boxes, scores, classes):
            results.append({
                "file": os.path.basename(file_path),
                "class_id": int(cls),
                "class_name": CATEGORY_NAMES[int(cls)],
                "score": float(score),
                "bbox_xmin": float(box[0]),
                "bbox_ymin": float(box[1]),
                "bbox_xmax": float(box[2]),
                "bbox_ymax": float(box[3]),
            })


        v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = v.draw_instance_predictions(instances)
        out_path = os.path.join(SAVE_DIR, os.path.basename(file_path))
        cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()