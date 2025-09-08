def main():
    import os
    import sys
    sys.path.insert(0, os.path.abspath('detectron2'))
    import cv2
    import numpy as np
    import pandas as pd
    from glob import glob
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    # --- Setup ---
    IMAGE_DIR = "Images/2048"
    COCO_JSON = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Models_Training/Detectron2/Aug_Enhanced_2x/test/_annotations.coco.json"
    SAVE_DIR = "predicted_results/Aug_Enhanced_2x/groundtruth_background/2048"
    PIXEL_SIZE_AT_1280 = 0.198595
    M2_TO_FT2 = 10.7639
    IMAGE_SIZE = 1280
    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
    HOUSE_CLASS_ID = 2
    GARAGE_CLASS_ID = 1

    CLASSES_TO_MEASURE = {
        HOUSE_CLASS_ID: "House",
        GARAGE_CLASS_ID: "Garage"
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    coco = COCO(COCO_JSON)
    metadata = MetadataCatalog.get("coco_gt")
    metadata.set(thing_classes=CATEGORY_NAMES)

    image_id_to_filename = {img['id']: img['file_name'] for img in coco.dataset['images']}
    image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png"))

    for img_id, img_info in coco.imgs.items():
        file_name = img_info["file_name"]
        file_path = os.path.join(IMAGE_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        image_np = cv2.imread(file_path)
        height, width = image_np.shape[:2]
        pixel_size_m = PIXEL_SIZE_AT_1280 * (IMAGE_SIZE / width)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        object_infos = []

        for ann in anns:
            cls_id = ann["category_id"]
            if cls_id not in CLASSES_TO_MEASURE:
                continue

            rle = coco.annToRLE(ann)
            mask = maskUtils.decode(rle)
            pixel_area = mask.sum()
            real_area_m2 = pixel_area * (pixel_size_m ** 2)
            real_area_ft2 = real_area_m2 * M2_TO_FT2

            x, y, w, h = ann["bbox"]
            box = [x, y, x + w, y + h]

            object_infos.append({
                "object_id": ann["id"],
                "class_id": int(cls_id),
                "class_name": CATEGORY_NAMES[int(cls_id)],
                "bbox": box,
                "area_m2": real_area_m2,
                "area_ft2": real_area_ft2,
                "file": file_name,
            })


        # --- Visualization ---
        vis = Visualizer(image_np[:, :, ::-1], metadata=metadata, scale=1.0)
        
        for obj in object_infos:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label = f"{obj['class_name']} ID:{obj['object_id']} Area:{obj['area_ft2']:.2f} ft2"
            color = (0, 255, 0) if obj['class_name'] == "House" else (255, 255, 66)

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        out_path = os.path.join(SAVE_DIR, file_name)
        cv2.imwrite(out_path, image_np)

        if object_infos:
            df = pd.DataFrame(object_infos)
            df.to_csv(f"{SAVE_DIR}/{file_name}_objects.csv", index=False)

if __name__ == "__main__":
    main()
