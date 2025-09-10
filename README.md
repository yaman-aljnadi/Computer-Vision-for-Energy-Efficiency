<div align="center">
  [** Models **] | [** [Setup](#Setup) **] 
  
  
</div>

# Describtion 
This projects looks into building and identifiing what is the best model that can be used for segmenting houses and similar objects from satellite images, there are three images and object detection models architicures used in this project along with 4 augemntaions techniques to generate the images, so in total we would have 12 trained models, and then we can analyze which one would be the best for segmentaiotn. This thing even will help us to look into how the models would behave with the augmentation, pre-processing, and enhancemnt. Also trying to identify how different architictures behaves with the different augmentations.

Generated Dataset Variants for Model Training
* Original 1280×1280 (color + augmentation)
* Original 1280×1280 (greyscale + augmentation)
* Enhanced 2048×2048 (color + augmentation)
* Enhanced 2048×2048 (greyscale + augmentation)

Models and Architectures Used
* YOLOv11: yolo11x-seg.pt
* Detectron2: Mask RCNN_X_101_32x8d_FPN_3x
* MMDetection: Mask RCNN_x101-64x4d_FPN_2x




# Setup
```Python 3.9.21```
```pip install requirments.txt```


# Computer-Vision-for-Energy-Efficiency
This project explores the use of computer vision techniques to improve energy efficiency analysis through satellite image segmentation. By using deep learning architectures, image enhancement methods, and data augmentation, we aim to evaluate how different dataset variants and model choices affect segmentation performance.


