import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import torch
import sys
import math
import random

#MMDetection Imports
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

#Detectron2 Imports
sys.path.insert(0, os.path.abspath('Detectron2/detectron2'))
print(torch.cuda.memory_summary())
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

def get_center(bbox):
    """Calculates the center (x, y) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(pt1, pt2):
    """Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def generate_color(id_val):
    """Generates a consistent random BGR color based on an ID."""
    random.seed(id_val)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def main():
    # MMDetection Config 
    MM_CONFIG_FILE = 'MMDetection/Aug_Original/mask_rcnn_custom_V2/mask_rcnn_custom.py'
    MM_CHECKPOINT_FILE = 'MMDetection/Aug_Original/mask_rcnn_custom_V2/epoch_20.pth'
    
    # Detectron2 Config 
    D2_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    D2_CHECKPOINT_PATH = "Detectron2/Aug_Greyscale_Enhanced_x2/output_detectron2/model_final.pth"
    
    # --- Directories ---
    IMG_DIR_1280 = 'Workflow_Images_Results/Images/Detectron2_Greyscale_Enahnced'
    IMG_DIR_2048 = "Workflow_Images_Results/Images/MMDetection"
    SAVE_DIR = 'Workflow_Images_Results/Results_All_Updated/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Classes
    CATEGORY_NAMES = ['None', 'Garage', 'House', 'Other', 'Propane', 'Trailer']
    
    # Target Definitions
    PRIMARY_CLASSES = ["House", "Trailer", "Other"] 
    SECONDARY_CLASSES = ["Garage", "Propane"] # Objects we want to link to the house

    D2_TARGET_CLASS_ID = 4  # Propane
    
    # Calculation 
    # PIXEL_SIZE = 0.226042119 # old
    PIXEL_SIZE = 0.198595 
    M2_TO_FT2 = 10.7639
    M_TO_FT = math.sqrt(M2_TO_FT2) 
    
    # Proximity Configuration 
    # We will calculate the threshold dynamically based on image size later
    PROXIMITY_THRESHOLD_PERCENT = 0.10 # 10% of image diagonal


    print("Initializing Models...")
    
    # Init MMDetection
    mm_model = init_detector(MM_CONFIG_FILE, MM_CHECKPOINT_FILE, device='cuda:0')
    
    # Init Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(D2_CONFIG_FILE))
    cfg.MODEL.WEIGHTS = D2_CHECKPOINT_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 2048
    cfg.INPUT.MAX_SIZE_TEST = 2048
    
    d2_predictor = DefaultPredictor(cfg)

    image_paths_1280 = glob(os.path.join(IMG_DIR_1280, "*.jpg")) + glob(os.path.join(IMG_DIR_1280, "*.png"))
    print(f"Found {len(image_paths_1280)} images to process.")

    for file_path_1280 in image_paths_1280:
        filename = os.path.basename(file_path_1280)
        print(f"Processing: {filename}")

        # Load Base Image (1280)
        img_1280 = cv2.imread(file_path_1280)
        if img_1280 is None:
            continue
            
        height_1280, width_1280 = img_1280.shape[:2]
        pixel_size_m = PIXEL_SIZE 
        
        # Calculate Safety Net Threshold  --> 10%
        img_diagonal = math.sqrt(width_1280**2 + height_1280**2)
        dist_threshold = img_diagonal * PROXIMITY_THRESHOLD_PERCENT

        # Load Enhanced Image (2048)
        file_path_2048 = os.path.join(IMG_DIR_2048, filename)
        img_2048 = cv2.imread(file_path_2048) if os.path.exists(file_path_2048) else None

        combined_objects = []
        
        #  Run MMDetection 
        mm_result = inference_detector(mm_model, img_1280)
        instances = mm_result.pred_instances
        
        mm_boxes = instances.bboxes.cpu().numpy()
        mm_scores = instances.scores.cpu().numpy()
        mm_classes = instances.labels.cpu().numpy()
        mm_masks = instances.masks.cpu().numpy() if hasattr(instances, 'masks') else None

        # Filter MM results
        keep_indices = []
        for i in range(len(mm_scores)):
            if mm_scores[i] >= 0.25 and mm_classes[i] != D2_TARGET_CLASS_ID:
                keep_indices.append(i)
        
        mm_boxes, mm_scores, mm_classes = mm_boxes[keep_indices], mm_scores[keep_indices], mm_classes[keep_indices]
        if mm_masks is not None: mm_masks = mm_masks[keep_indices]

        # IoU Filtering
        keep_iou = list(range(len(mm_boxes)))
        for i in range(len(mm_boxes)):
            for j in range(i + 1, len(mm_boxes)):
                if i not in keep_iou or j not in keep_iou: continue
                if mm_classes[i] == mm_classes[j]: continue 

                mask_i = mm_masks[i].astype(bool)
                mask_j = mm_masks[j].astype(bool)
                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()
                
                if union > 0 and (intersection/union) > 0.7: 
                    if mm_scores[i] >= mm_scores[j]:
                        if j in keep_iou: keep_iou.remove(j)
                    else:
                        if i in keep_iou: keep_iou.remove(i)
        
        # Store MM Objects
        vis_img = img_1280.copy()
        alpha = 0.2
        
        # Temp storage to hold masks for drawing later
        object_masks = {} 

        for idx in keep_iou:
            box = mm_boxes[idx]
            score = mm_scores[idx]
            cls_id = int(mm_classes[idx])
            mask = mm_masks[idx]
            class_name = CATEGORY_NAMES[cls_id]
            
            pixel_area = mask.sum()
            real_area_ft2 = (pixel_area * (pixel_size_m ** 2)) * M2_TO_FT2
            
            obj_id = len(combined_objects)
            combined_objects.append({
                "id": obj_id,
                "source": "MMDet",
                "class_id": cls_id,
                "class_name": class_name,
                "score": float(score),
                "area_ft2": real_area_ft2,
                "bbox": box.tolist(),
                "center": get_center(box),
                "parent_id": None, 
                "color": None,
                "dist_px": None # Placeholder for distance
            })
            object_masks[obj_id] = mask 

        # Run Detectron2 
        if img_2048 is not None:
            h_2048, w_2048 = img_2048.shape[:2]
            scale_x = width_1280 / float(w_2048)
            scale_y = height_1280 / float(h_2048)

            d2_outputs = d2_predictor(img_2048)
            d2_instances = d2_outputs["instances"].to("cpu")
            d2_boxes = d2_instances.pred_boxes.tensor.numpy()
            d2_scores = d2_instances.scores.numpy()
            d2_classes = d2_instances.pred_classes.numpy()

            for i, (box, score, cls_id) in enumerate(zip(d2_boxes, d2_scores, d2_classes)):
                if int(cls_id) == D2_TARGET_CLASS_ID and score >= 0.25:
                    x1, y1, x2, y2 = box
                    x1_s = max(0, min(width_1280 - 1, x1 * scale_x))
                    y1_s = max(0, min(height_1280 - 1, y1 * scale_y))
                    x2_s = max(0, min(width_1280 - 1, x2 * scale_x))
                    y2_s = max(0, min(height_1280 - 1, y2 * scale_y))
                    
                    scaled_box = [x1_s, y1_s, x2_s, y2_s]
                    
                    combined_objects.append({
                        "id": len(combined_objects),
                        "source": "Detectron2",
                        "class_id": int(cls_id),
                        "class_name": "Propane",
                        "score": float(score),
                        "area_ft2": 0.0, # As per original code, D2 area is 0 here
                        "bbox": scaled_box,
                        "center": get_center(scaled_box),
                        "parent_id": None,
                        "color": None,
                        "dist_px": None
                    })

        # OBJECT ASSOCIATION

        # Identify Primary Structures (House, Trailer, Other)
        primary_structures = [obj for obj in combined_objects if obj["class_name"] in PRIMARY_CLASSES]
        
        # Assign a unique color to each primary structure
        for i, struct in enumerate(primary_structures):
            struct_color = generate_color(i * 100)
            struct["color"] = struct_color
            struct["house_idx"] = i + 1 

        # Connect Secondary Objects (Garage, Propane)
        for obj in combined_objects:
            if obj["class_name"] in SECONDARY_CLASSES:
                
                min_dist = float('inf')
                closest_struct = None
                
                # Search for closest primary structure
                for struct in primary_structures:
                    dist = calculate_distance(obj["center"], struct["center"])
                    if dist < min_dist:
                        min_dist = dist
                        closest_struct = struct
                
                # Check Safety Net (10% threshold)
                if closest_struct is not None and min_dist <= dist_threshold:
                    # Link them
                    obj["parent_id"] = closest_struct["id"]
                    obj["color"] = closest_struct["color"] 
                    obj["house_idx"] = closest_struct["house_idx"]
                    obj["dist_px"] = min_dist 
                else:
                    # Too far away, or no structure found
                    obj["color"] = (128, 128, 128) 
                    obj["house_idx"] = None
                    obj["dist_px"] = None

        # Final Drawing & Saving 
        
        # Draw Masks
        for obj in combined_objects:
            if obj["id"] in object_masks:
                mask = object_masks[obj["id"]]
                color = obj["color"] if obj["color"] is not None else (0, 255, 0)
                
                mask_indices = mask.astype(bool)
                color_array = np.full((*mask.shape, 3), color, dtype=np.uint8)
                vis_img[mask_indices] = cv2.addWeighted(color_array[mask_indices], alpha, vis_img[mask_indices], 1 - alpha, 0)

        # Draw Boxes, Labels, and Connection Lines
        for obj in combined_objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cx, cy = map(int, obj["center"])
            
            color = obj["color"]
            if color is None:
                color = (0, 0, 255) if obj["class_name"] == "Propane" else (0, 255, 0)

            # Draw Connection Line if linked
            if obj["parent_id"] is not None:
                parent = next((item for item in combined_objects if item["id"] == obj["parent_id"]), None)
                if parent:
                    px, py = map(int, parent["center"])
                    cv2.line(vis_img, (cx, cy), (px, py), color, 2)
                    cv2.circle(vis_img, (cx, cy), 5, color, -1)

            # Final Drawing & Saving loop

            # Class Name + ID 
            if "house_idx" in obj and obj["house_idx"] is not None:
                label_main = f"{obj['class_name']} (ID:{obj['house_idx']})"
            else:
                label_main = f"{obj['class_name']}"

            # Stats (Area + Distance)
            stats_parts = []

            # Area: Only calculate/show if it is NOT a Propane tank
            if obj["class_name"] != "Propane":
                stats_parts.append(f"{obj['area_ft2']:.0f} sqft")

            # Distance: Only show if a distance exists (it is linked)
            if obj["dist_px"] is not None:
                dist_ft = obj["dist_px"] * pixel_size_m * M_TO_FT
                stats_parts.append(f"Dist: {dist_ft:.1f} ft")
            
            # separator when adding text
            label_stats = " | ".join(stats_parts)

            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Draw Text Stacked
            # Draw Main Label (Top)
            cv2.putText(
                vis_img, label_main, (x1, max(y1 - 28, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )
            
            # Draw Stats Label (Bottom) - Only if we have stats to show
            if label_stats:
                cv2.putText(
                    vis_img, label_stats, (x1, max(y1 - 8, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                )

        # Save Image
        out_img_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(out_img_path, vis_img)
        
        # Save CSV data 
        csv_data = []
        for obj in combined_objects:
            entry = {
                "class_name": obj["class_name"],
                "score": obj["score"],
                "area_ft2": obj["area_ft2"],
                "linked_house": obj.get("house_idx", "None"),
                "distance_from_house_px": "N/A",
                "distance_from_house_ft": "N/A"
            }
            
            if obj["dist_px"] is not None:
                 entry["distance_from_house_px"] = f"{obj['dist_px']:.1f}"
                 # Add feet to CSV as well for convenience
                 dist_ft = obj["dist_px"] * pixel_size_m * M_TO_FT
                 entry["distance_from_house_ft"] = f"{dist_ft:.1f}"
            
            csv_data.append(entry)

        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(SAVE_DIR, f"{filename}_objects.csv"), index=False)

    print("Done! Check Combined_Results folder.")

if __name__ == "__main__":
    main()