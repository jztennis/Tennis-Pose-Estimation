from ultralytics import YOLO

# model
model = YOLO("yolov8n-pose.yaml")
model.train(
    data="data.yaml",
    epochs=10,
    imgsz=1080,
    batch=16,
    name="tennis_pose_model"
)
