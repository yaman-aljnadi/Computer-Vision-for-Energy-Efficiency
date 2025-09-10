<div align="center">

  # [**Models**](Docs/Models.md) | [**Results**](#Results)
  
</div>


# Describtion 
The Computer Vision for Energy Efficiency project looks into building a Segmentation AI model that is cabaple of accuratly segmentaing Houses and similar objects using satellite images for size calculation and electrical power analysis. 



# More in Details
470 labaled satellite images are used with the following objects 
- Houses
- Garages
- Trailers
- Propanes
- Others

In order to find the best model the following data preprocessing and augmentation have been used: The augmentations that have been done are the following to the images: Vertical flips, horizintal flips and Greyscale. Also we have used an enhancment AI model called [ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)


Here are the Generated Dataset Variants for Model Training
* Original 1280×1280 (color + augmentation)
* Original 1280×1280 (greyscale + augmentation)
* Enhanced 2048×2048 (color + augmentation)
* Enhanced 2048×2048 (greyscale + augmentation)

And in terms of models architicures here are the models that have been used
Models and Architectures Used
* YOLOv11: yolo11x-seg.pt
* Detectron2: Mask RCNN_X_101_32x8d_FPN_3x
* MMDetection: Mask RCNN_x101-64x4d_FPN_2x

So we should have 12 models in general, but unfortunatly duo to the lack of computional resources we couldn't train the Enhanced images version with the MMDetection so that leaves us with only 10 models. 

# Results
Excell Results including Percision, Recall, mAP, ...etc. [[here]](Logs/Models_Results/) </br>
Confusion Matrix of all models. [[here]](Logs/Models_Results/Results.md)


# Computer-Vision-for-Energy-Efficiency
This project explores the use of computer vision techniques to improve energy efficiency analysis through satellite image segmentation. By using deep learning architectures, image enhancement methods, and data augmentation, we aim to evaluate how different dataset variants and model choices affect segmentation performance.


