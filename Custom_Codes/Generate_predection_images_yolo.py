from ultralytics import YOLO

model = YOLO("/home/data/files/Yaman/Satalite_Images/Models_Training/Yolov11_Seg/Aug_Greyscale_Enhanced_x2/yolo_outputs_all/yolov11_seg/weights/best.pt")

results = model.predict(source="/home/data/files/Yaman/Satalite_Images/Models_Training/Yolov11_Seg/Aug_Greyscale_Enhanced_x2/dataset/test/images/", save=True, imgsz=2048, task='segment', device=0)