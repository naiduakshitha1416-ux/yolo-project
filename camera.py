from ultralytics import YOLO

# load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# start webcam detection
model.predict(source=0, show=True)