<div align="center">

# Computer Vision for Energy Efficiency  
**Satellite Image Segmentation for Houses, Garages, Trailers & Propane Tanks**

[**Models**](Docs/Models.md) • [**Full Results**](Docs/Results.md) • [**Logs**](Logs/Models_Results/)

</div>

## Overview
This project builds and compares instance/semantic segmentation models to accurately segment houses and related structures in satellite imagery. The goal is to enable reliable **roof area estimation** for downstream **electrical load and energy-efficiency analysis**.

## Dataset
- **Images:** 470 labeled satellite images  
- **Classes:** `House`, `Garage`, `Trailer`, `Propane`, `Other`
- **Variants (generated):**
  - Original **1280×1280** color (+ augmentation)
  - Original **1280×1280** grayscale (+ augmentation)
  - ESRGAN-enhanced **2048×2048** color (+ augmentation)
  - ESRGAN-enhanced **2048×2048** grayscale (+ augmentation)

**Augmentations:** horizontal/vertical flips, grayscale; (roadmap adds rotations, color jitter, blur, cutout).  
**Enhancement:** [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)

## Models & Frameworks
We target 12 combinations (3 frameworks × 2 input sizes × color/grayscale).  
Due to compute limits, MMDetection on 2048 variants is pending (10 models trained so far).

| Framework    | Architecture / Checkpoint                       | Input | Color | Status |
|--------------|--------------------------------------------------|-------|-------|--------|
| YOLOv11      | `yolo11x-seg.pt`                                 | 1280  | RGB   | ✅     |
| YOLOv11      | `yolo11x-seg.pt`                                 | 1280  | Gray  | ✅     |
| YOLOv11      | `yolo11x-seg.pt`                                 | 2048  | RGB   | ✅     |
| YOLOv11      | `yolo11x-seg.pt`                                 | 2048  | Gray  | ✅     |
| Detectron2   | Mask R-CNN `X_101_32x8d_FPN_3x`                  | 1280  | RGB   | ✅     |
| Detectron2   | Mask R-CNN `X_101_32x8d_FPN_3x`                  | 1280  | Gray  | ✅     |
| Detectron2   | Mask R-CNN `X_101_32x8d_FPN_3x`                  | 2048  | RGB   | ✅     |
| Detectron2   | Mask R-CNN `X_101_32x8d_FPN_3x`                  | 2048  | Gray  | ✅     |
| MMDetection  | Mask R-CNN `x101-64x4d_FPN_2x`                   | 1280  | RGB   | ✅     |
| MMDetection  | Mask R-CNN `x101-64x4d_FPN_2x`                   | 1280  | Gray  | ✅     |
| MMDetection  | Mask R-CNN `x101-64x4d_FPN_2x`                   | 2048  | RGB   | ⏳     |
| MMDetection  | Mask R-CNN `x101-64x4d_FPN_2x`                   | 2048  | Gray  | ⏳     |

> Detailed configs per framework live in [`configs/`](configs/).

## Results (summary)
A full breakdown (precision, recall, mAP50, mAP50-95, per-class IoU, confusion matrices) is in  
👉 [`Docs/Results.md`](Docs/Results.md) and raw logs in [`Logs/Models_Results/`](Logs/Models_Results/).

| Model (short) | Input | Color | mAP50-95 | mAP50 | Precision | Recall | Mean IoU | Notes |
|---|---:|:---:|---:|---:|---:|---:|---:|---|
| D2-X101-FPN | 2048 | RGB | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | best recall on Propane |
| YOLO11x-seg | 1280 | RGB | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | fastest inference |
| MMDet-X101 | 1280 | Gray | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx | robust to lighting |

> Replace `0.xx` with your numbers or auto-render from `experiments/runs.csv`.

## Reproducing Results

### Environment
```bash
# Option A: conda
conda env create -f environment.yml
conda activate cv-energy

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
