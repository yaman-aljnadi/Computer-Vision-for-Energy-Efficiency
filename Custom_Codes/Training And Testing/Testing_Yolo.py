from ultralytics import YOLO


model = YOLO("/home/data/files/Yaman/Satalite_Images/Models_Training/Yolov11_Seg/Aug_Original/yolo_outputs_all/yolov11_seg/weights/best.pt")

model.overrides['batch'] = 2


metrics = model.val(
    data="dataset/data.yaml", 
    imgsz=1280,
    task='segment',                     
    device=0,
    save_json=True,
    conf=0.25,
    split='test',             
)

print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)

