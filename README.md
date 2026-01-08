<div align="center">

# A Vision-Based Framework for Extracting Building Stock Characteristics from Satellite Imagery
### Supporting Energy Modeling in Data-Sparse Regions

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](A_Vision_Based_Framework_for_Extracting_Building_Stock_Characteristics_from_Satellite_Imagery_to_Support_Energy_Modeling.docx.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Frameworks](https://img.shields.io/badge/YOLOv11-Detectron2-MMDetection-blue)](Docs/Models.md)

[**Methodology**](#methodology) • [**Key Results**](#results) • [**Installation**](#installation) • [**Citation**](#citation)

</div>

---

## 📖 Overview
 This repository contains the official implementation and data for the study: **"A Vision-Based Framework for Extracting Building Stock Characteristics from Satellite Imagery to Support Energy Modeling."**

Building Stock Energy Modeling (BSEM) often suffers from data gaps, particularly in rural areas where assessor records are incomplete. This project utilizes deep learning and satellite imagery to automatically extract key building characteristics required for energy modeling:
1.  **Building Footprint Area:** Via instance segmentation.
2.  **Building Type:** Categorizing Single-Family vs. Manufactured Homes.
3.  **Heating Fuel Proxy:** Detecting external propane tanks to infer non-gas heating.
4.  **Garage Presence:** Associating detached garages with the nearest home.

## 🖼️ Dataset
The dataset focuses on rural and cold-climate regions of the northern United States, specifically tailored for BSEM applications.

* **Source:** Google Maps Satellite Tiles (Zoom Level 18).
* **Size:** 490 labeled images ($1280 \times 1280$), augmented to **1,100 images**.
* **Classes:** `Single-Family Home`, `Manufactured Home (Trailer)`, `Detached Garage`, `Propane Tank`.
* **Enhancements:** * **Grayscale Conversion:** To test computational efficiency.
    * **ESRGAN (Super-Resolution):** Upscaling to $2048 \times 2048$ to improve small object detection (propane tanks).

## 🏗️ Methodology & Models
We evaluated three state-of-the-art frameworks across four dataset variants (Original RGB, Grayscale, ESRGAN RGB, ESRGAN Grayscale).

### The Pipeline
The project implements an integrated workflow that combines segmentation, detection, and spatial association:
1.  **Segmentation:** Extract building masks and calculate real-world footprint area ($m^2$).
2.  **Detection:** Identify propane tanks (small objects).
3.  **Integration:** Use Euclidean distance to associate tanks and garages with the nearest residential structure.

![Workflow Pipeline](Docs/assets/workflow_pipeline.png)
*(Figure 5 from the paper: The integrated data extraction pipeline)*

### Model Configurations Evaluated

| Framework | Architecture | Task | Best For |
| :--- | :--- | :--- | :--- |
| **YOLOv11** | `yolo11x-seg` | Unified | Fast inference, balanced performance. |
| **Detectron2** | Mask R-CNN `X_101_32x8d_FPN_3x` | Detection | **Best Propane Detection** (Small objects). |
| **MMDetection** | Mask R-CNN `x101-64x4d_FPN_2x` | Segmentation | **Best Building Segmentation** & Footprint estimation. |

> **Note:** MMDetection training on ESRGAN ($2048 \times 2048$) images was omitted due to GPU memory constraints.

## 📊 Results

### 1. Propane Tank Detection (Heating Fuel Proxy)
Propane tanks are small and difficult to detect. We found that **Image Super-Resolution (ESRGAN)** combined with **Grayscale** inputs yielded the highest accuracy using Detectron2.

| Model | Input | Precision | Recall | F1-Score | mAP 50 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Detectron2** | **Grayscale + ESRGAN** | **0.828** | **0.930** | **0.876** | **0.894** |
| YOLOv11 | ESRGAN | 0.934 | 0.750 | 0.832 | 0.894 |
| Detectron2 | Grayscale (Low Compute) | 0.888 | 0.828 | 0.857 | 0.802 |

### 2. Building Segmentation & Categorization
For building footprints, **MMDetection** on original resolution imagery outperformed super-resolution techniques.

| Model | Input | Precision | Recall | F1-Score | mAP 50 | mAP 50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MMDetection** | **Original RGB** | **0.601** | **0.718** | **0.654** | **0.741** | **0.641** |
| Detectron2 | ESRGAN | 0.542 | 0.762 | 0.633 | 0.734 | 0.637 |
| YOLOv11 | ESRGAN | 0.642 | 0.679 | 0.660 | 0.648 | 0.572 |

* **Single-Family Homes:** The best model achieved an **mAP50 of 0.955** for this specific class.
* **Footprint Accuracy:** Comparison with assessor data for 20 homes showed an average error of only **5.46%**.

### Visual Results
![Visual Results](Docs/assets/results_example.png)
*(Figure 6 from the paper: Integrated outputs showing segmentation, propane detection, and association in diverse densities)*

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch (CUDA recommended)

### Setup
```bash
# Clone the repository
git clone [https://github.com/yaman-aljnadi/Computer-Vision-for-Energy-Efficiency.git](https://github.com/yaman-aljnadi/Computer-Vision-for-Energy-Efficiency.git)
cd Computer-Vision-for-Energy-Efficiency

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
