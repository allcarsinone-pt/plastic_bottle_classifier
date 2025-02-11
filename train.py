from ultralytics import YOLO

# Init yolo model
model = YOLO("yolov8n.pt")

results = model.train(data="./data.yaml", epochs=35)

results = model.val()


success = model.export(format="onnx")

