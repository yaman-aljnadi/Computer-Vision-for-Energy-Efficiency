import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

def main():

    CONFIG_FILE = 'H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Logs/Seg_Logs/MMDetection/Aug_Original/mask_rcnn_custom.py'
    CHECKPOINT_FILE = 'H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Logs/Seg_Logs/MMDetection/Aug_Original/epoch_20.pth'
    IMAGE_DIR = 'Images/Custome_Collected_Adnan_PDF'
    SAVE_DIR = 'predicted_results/Adnan_PDF_Results'
    os.makedirs(SAVE_DIR, exist_ok=True)

    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
    PIXEL_SIZE = 0.198595
    M2_TO_FT2 = 10.7639
    IMAGE_SIZE = 1280

    HOUSE_CLASS_ID = 2
    GARAGE_CLASS_ID = 1
    TRAILER_CLASS_ID = 5

    CLASSES_TO_MEASURE = {
        HOUSE_CLASS_ID: "House",
        GARAGE_CLASS_ID: "Garage",
        TRAILER_CLASS_ID: "Trailer"
    }

    # Initialize model
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    visualizer = DetLocalVisualizer()

    image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob(os.path.join(IMAGE_DIR, "*.png"))

    results_all = []

    for file_path in image_paths:

        print(f"Processing {file_path}")

        image_np = cv2.imread(file_path)
        height, width = image_np.shape[:2]

        # pixel_size_m = PIXEL_SIZE * (IMAGE_SIZE / width)
        pixel_size_m = PIXEL_SIZE 

        # print(f"Original size: {width}x{height}")
        # print(f"Pixel size: {pixel_size_m} m/pixel")

        # expected_pixels = (1664 / M2_TO_FT2) / (pixel_size_m ** 2)
        # print(f"Expected pixel count: {expected_pixels}")


        result = inference_detector(model, image_np)

        temp_vis_path = os.path.join(SAVE_DIR, 'temp_vis.jpg')

        visualizer.add_datasample(
            name='result',
            image=image_np,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=temp_vis_path,
            pred_score_thr=0.25
        )

        instances = result.pred_instances

        boxes = instances.bboxes.cpu().numpy() if instances.bboxes is not None else np.empty((0, 4))
        scores = instances.scores.cpu().numpy() if instances.scores is not None else np.array([])
        classes = instances.labels.cpu().numpy() if instances.labels is not None else np.array([])
        masks = None
        if hasattr(instances, 'masks') and instances.masks is not None:
            masks = instances.masks.cpu().numpy()
        else:
            masks = np.zeros((len(boxes), height, width), dtype=bool)

        # Thresholding
        keep = scores >= 0.25
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        masks = masks[keep]


        # IoU-based filtering between different classes
        keep_indices = list(range(len(boxes)))

        IOU_THRESHOLD = 0.7
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if i not in keep_indices or j not in keep_indices:
                    continue

                if classes[i] == classes[j]:
                    continue

                mask_i = masks[i].astype(bool)
                mask_j = masks[j].astype(bool)

                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()

                if union == 0:
                    continue

                iou = intersection / union
                if iou > IOU_THRESHOLD:
                    if scores[i] >= scores[j]:
                        keep_indices.remove(j)
                    else:
                        keep_indices.remove(i)

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
        masks = masks[keep_indices]

        # Create object infos
        object_infos = []
        for idx, (box, score, cls_id, mask) in enumerate(zip(boxes, scores, classes, masks)):
            if cls_id in CLASSES_TO_MEASURE:
                pixel_area = mask.sum()

                # print(f"Predicted pixel count: {mask.sum()}")
                # print(cv2.countNonZero(mask.astype(np.uint8)))

                real_area_m2 = pixel_area * (pixel_size_m ** 2)
                real_area_ft2 = real_area_m2 * M2_TO_FT2

                object_infos.append({
                    "object_id": idx,
                    "class_id": int(cls_id),
                    "class_name": CATEGORY_NAMES[int(cls_id)],
                    "bbox": box.tolist(),
                    "area_m2": real_area_m2,
                    "area_ft2": real_area_ft2,
                    "score": float(score),
                    "file": os.path.basename(file_path),
                })

            results_all.append({
                "file": os.path.basename(file_path),
                "class_id": int(cls_id),
                "class_name": CATEGORY_NAMES[int(cls_id)],
                "score": float(score),
                "bbox_xmin": float(box[0]),
                "bbox_ymin": float(box[1]),
                "bbox_xmax": float(box[2]),
                "bbox_ymax": float(box[3]),
            })

        vis_img_mm = cv2.imread(temp_vis_path)
        if vis_img_mm is None:
            vis_img_mm = image_np.copy()
        else:
            vis_img_mm = vis_img_mm[:, :, ::-1]

        vis_img = vis_img_mm.copy()
        alpha = 0.2  
        overlay = vis_img.copy()

        for idx, (mask, cls_id) in enumerate(zip(masks, classes)):
            color = (0, 255, 0) if cls_id == HOUSE_CLASS_ID else (255, 255, 66) 
            color_array = np.full((*mask.shape, 3), color, dtype=np.uint8)

            mask_indices = mask.astype(bool)
            overlay[mask_indices] = cv2.addWeighted(color_array[mask_indices], alpha, vis_img[mask_indices], 1 - alpha, 0)

        vis_img = overlay

        for info in object_infos:
            box = info["bbox"]
            object_id = info["object_id"]
            class_name = info["class_name"]
            area = info["area_ft2"]

            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} ID:{object_id} Area:{area:.2f} ft2"

            color = (0, 255, 0) if class_name == "House" else (255, 255, 66)

            # cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                vis_img,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                1,
                cv2.LINE_AA
            )

        out_path = os.path.join(SAVE_DIR, os.path.basename(file_path))
        cv2.imwrite(out_path, vis_img)

        if object_infos:
            object_df = pd.DataFrame(object_infos)
            object_df.to_csv(f"{SAVE_DIR}/{os.path.basename(file_path)}_objects.csv", index=False)

    print("Done!")

if __name__ == "__main__":
    main()
