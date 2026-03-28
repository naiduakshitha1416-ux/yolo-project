from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="data.yaml", epochs=50, imgsz=640)