from ultralytics import YOLO

yaml_file = "yolo11/data.yaml"  
output_dir = "yolo_outputs_all"
run_name = "yolo11/yolov11"

model = YOLO("yolov11x.pt")
model.train(
    data=yaml_file,
    project=output_dir,
    name=run_name,
    epochs=50,
    workers=8,
    batch=16,
    device=0,
    save=True,
    verbose=True,
    exist_ok=True,
)