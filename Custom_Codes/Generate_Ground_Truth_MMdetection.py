import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

def coco_segmentation_to_mask(segmentation, height, width):
    rle = maskUtils.frPyObjects(segmentation, height, width)
    rle = maskUtils.merge(rle)
    return maskUtils.decode(rle)

def main():
    ANNOTATION_FILE = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/MMDetection/Aug_Original/test/_annotations.coco.json"
    IMAGE_DIR = "Images/1280"
    SAVE_DIR = "ground_truth_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    PIXEL_SIZE = 0.198595  # meters per pixel at 1280 resolution
    M2_TO_FT2 = 10.7639

    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
    CLASSES_TO_MEASURE = {1: "Garage", 2: "House"}  # Update with actual category_id from COCO

    # Initialize COCO
    coco = COCO(ANNOTATION_FILE)

    results_all = []

    for image_info in coco.loadImgs(coco.getImgIds()):
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        pixel_size_m = PIXEL_SIZE * (1280 / width)

        image_path = os.path.join(IMAGE_DIR, file_name)
        image_np = cv2.imread(image_path)

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        object_infos = []

        for idx, ann in enumerate(anns):
            cls_id = ann['category_id']
            cls_name = coco.loadCats(cls_id)[0]['name']

            mask = coco_segmentation_to_mask(ann['segmentation'], height, width)

            # Set default values
            area_m2 = None
            area_ft2 = None
            label = f"{cls_name} ID:{idx}"

            # Only calculate area for selected classes
            if cls_id in CLASSES_TO_MEASURE:
                pixel_area = mask.sum()
                area_m2 = pixel_area * (pixel_size_m ** 2)
                area_ft2 = area_m2 * M2_TO_FT2
                label += f" Area:{area_ft2:.2f} ft2"

                object_infos.append({
                    "object_id": idx,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "area_m2": area_m2,
                    "area_ft2": area_ft2,
                    "file": file_name,
                })

            # Define color (customize as needed)
            color = (0, 255, 0) if cls_name == "House" else (255, 255, 66) if cls_name == "Garage" else (200, 100, 255)

            # Overlay mask only for measured classes (optional)
            if cls_id in CLASSES_TO_MEASURE:
                overlay = image_np.copy()
                overlay[mask == 1] = (
                    0.4 * np.array(color) + 0.6 * overlay[mask == 1]
                ).astype(np.uint8)
                image_np = overlay

            # Draw bounding box
            bbox = ann["bbox"]
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

            # Put label
            cv2.putText(
                image_np,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        out_path = os.path.join(SAVE_DIR, file_name)
        cv2.imwrite(out_path, image_np)

        if object_infos:
            object_df = pd.DataFrame(object_infos)
            object_df.to_csv(f"{SAVE_DIR}/{file_name}_gt_objects.csv", index=False)
            results_all.extend(object_infos)

    print("Done!")

if __name__ == "__main__":
    main()
