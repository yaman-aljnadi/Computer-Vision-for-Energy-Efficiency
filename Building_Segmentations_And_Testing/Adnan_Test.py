
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

    DATASET_NAME = "satellite_val"
    # COCO_JSON = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Detectron2_Seg_aug_Original/test/_annotations.coco.json"
    # IMAGE_DIR = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Aug_Enhanced_2x/test/"
    CHECKPOINT_PATH = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Logs/Seg_Logs/Detectron2/Detectron2_Seg_aug_Original/model_final.pth"
    CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    RESULT_CSV = "Aug_Enhanced_2x/MaskRCNN_segmentation_results.csv"
    CATEGORY_NAMES = ['None','Garage', 'House', 'Other', 'Propane', 'Trailer']

    setup_logger(output="output", name="detectron2")

    datasets = ['satellite_val']

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    CHECKPOINT_DIR = "Aug_Enhanced_2x/MaskRCNN_segmentation_results"

    CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    Results = {}

    writer = SummaryWriter(log_dir="output")

    for dataset in datasets:
        print(f"Testing on dataset: {dataset}")

        val_json = f"H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Aug_Enhanced_2x/test/_annotations.coco.json"
        val_images = f"H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Aug_Enhanced_2x/test/"

        val_name = f"{dataset}_val"
        
        if val_name not in DatasetCatalog.list():
            register_coco_instances(val_name, {}, val_json, val_images)

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


        # save_dir = "Aug_Enhanced_2x/Adnan_Test"
        # os.makedirs(save_dir, exist_ok=True)

        # results = []
        # metadata = MetadataCatalog.get(val_name)

        # for i, data in enumerate(build_detection_test_loader(cfg, val_name)):
        #     file_name = data[0]["file_name"]
        #     image_np = cv2.imread(file_name)  # Load as uint8 BGR (as expected by predictor)

        #     outputs = predictor(image_np)
        #     instances = outputs["instances"].to("cpu")

        #     boxes = instances.pred_boxes.tensor.numpy()
        #     scores = instances.scores.numpy()
        #     classes = instances.pred_classes.numpy()
            
        #     for box, score, cls in zip(boxes, scores, classes):
        #         results.append({
        #             "file": os.path.basename(file_name),
        #             "class_id": int(cls),
        #             "class_name": CATEGORY_NAMES[int(cls)],  # +1 for background offset
        #             "score": float(score),
        #             "bbox_xmin": float(box[0]),
        #             "bbox_ymin": float(box[1]),
        #             "bbox_xmax": float(box[2]),
        #             "bbox_ymax": float(box[3]),
        #         })

        #     # Visualize and save
        #     v = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)  # BGR → RGB
        #     vis = v.draw_instance_predictions(instances)
        #     out_path = os.path.join(save_dir, os.path.basename(file_name))
        #     cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])  # RGB → BGR for saving

        #     if instances.has("pred_masks"):
        #         masks = instances.pred_masks.cpu().numpy()
        #         combined_mask = masks.any(axis=0).astype("uint8")
        #         segmented_image = image_np.copy()
        #         segmented_image[combined_mask == 0] = 0
        #         masked_out_path = os.path.join(save_dir, "segmented_" + os.path.basename(file_name))
        #         cv2.imwrite(masked_out_path, segmented_image)


if __name__ == "__main__":
    main()