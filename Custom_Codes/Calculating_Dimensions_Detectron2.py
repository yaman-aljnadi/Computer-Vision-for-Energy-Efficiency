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
    import numpy as np
    from detectron2.structures import Instances, Boxes

    DATASET_NAME = "satellite_val"
    IMAGE_DIR = "Images/Custome_Collected_Adnan_PDF/2048"  
    COCO_JSON = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Aug_Original/test/_annotations.coco.json"
    CHECKPOINT_PATH = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Logs/Seg_Logs/Detectron2/Aug_Original/model_final.pth"
    CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    SAVE_DIR = "predicted_results/Test/2048"
    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']

    PIXEL_SIZE_AT_1280 = 0.2072
    M2_TO_FT2 = 10.7639

    IMAGE_SIZE = 1280
    HOUSE_CLASS_ID = 2
    GARAGE_CLASS_ID = 1

    CLASSES_TO_MEASURE = {
        HOUSE_CLASS_ID: "House",
        GARAGE_CLASS_ID: "Garage"
    }

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
    cfg.INPUT.MIN_SIZE_TEST = 1280
    cfg.INPUT.MAX_SIZE_TEST = 1280

    predictor = DefaultPredictor(cfg)


    metadata = MetadataCatalog.get(DATASET_NAME) 
    metadata.set(thing_classes=CATEGORY_NAMES)
    results = []


    for file_path in image_paths:
        print(f"Processing: {file_path}")
        image_np = cv2.imread(file_path)
        height, width = image_np.shape[:2]
        pixel_size_m = PIXEL_SIZE_AT_1280 * (IMAGE_SIZE / width)
        print(f"Image size: {width}x{height}, pixel size: {pixel_size_m} m/px")

        outputs = predictor(image_np)
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy()

        keep_indices = list(range(len(instances)))

        # IoU threshold beyond which we consider two masks overlapping
        IOU_THRESHOLD = 0.7

        # Compare all pairs
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                if i not in keep_indices or j not in keep_indices:
                    continue

                # Only check between different classes
                if classes[i] == classes[j]:
                    continue

                mask_i = masks[i]
                mask_j = masks[j]

                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()

                if union == 0:
                    continue

                iou = intersection / union

                if iou > IOU_THRESHOLD:
                    # Remove the one with the lower score
                    if scores[i] >= scores[j]:
                        keep_indices.remove(j)
                    else:
                        keep_indices.remove(i)

        # Keep only filtered instances
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
        masks = masks[keep_indices]

        object_infos = []

        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if cls in CLASSES_TO_MEASURE:
                mask = masks[i]
                pixel_area = mask.sum()
                real_area_m2 = pixel_area * (pixel_size_m ** 2)
                real_area_ft2 = real_area_m2 * M2_TO_FT2

                object_infos.append({
                    "object_id": i,
                    "class_id": int(cls),
                    "class_name": CATEGORY_NAMES[int(cls)],
                    "bbox": box,
                    "area_m2": real_area_m2,
                    "area_ft2": real_area_ft2,
                    "score": float(score),
                    "file": os.path.basename(file_path),
                })

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

        
        filtered_instances = Instances(image_size=(height, width))
        filtered_instances.pred_boxes = Boxes(torch.tensor(boxes))
        filtered_instances.scores = torch.tensor(scores)
        filtered_instances.pred_classes = torch.tensor(classes)
        filtered_instances.pred_masks = torch.tensor(masks)

        v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = v.draw_instance_predictions(filtered_instances)
        vis_image = vis.get_image()[:, :, ::-1].copy()

        for info in object_infos:
            box = info["bbox"]
            object_id = info["object_id"]
            class_name = info["class_name"]
            area = info["area_ft2"]

            x1, y1, x2, y2 = box.astype(int)
            label = f"{class_name} ID:{object_id} Area:{area:.2f} ft2"

            color = (0, 255, 0) if class_name == "House" else (255, 255, 66)

            cv2.rectangle(
                vis_image,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=2
            )

            cv2.putText(
                vis_image,
                label,
                org=(x1, max(y1 - 10, 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

        out_path = os.path.join(SAVE_DIR, os.path.basename(file_path))
        cv2.imwrite(out_path, vis_image)

        if object_infos:
            object_df = pd.DataFrame(object_infos)
            object_df.to_csv(f"{SAVE_DIR}/{os.path.basename(file_path)}_objects.csv", index=False)

if __name__ == "__main__":
    main()