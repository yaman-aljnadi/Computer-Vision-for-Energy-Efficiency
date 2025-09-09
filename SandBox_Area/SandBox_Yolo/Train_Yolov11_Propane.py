from ultralytics import YOLO

yaml_file = "yolov11_Propane/data.yaml"  
output_dir = "yolo_outputs_propane"
run_name = "yolov11_Propane/yolov11x_Propane"

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