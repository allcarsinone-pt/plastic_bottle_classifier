from ultralytics import YOLO
from PIL import Image
model = YOLO("./runs/detect/train7/weights/best.pt")  # load a custom model


uri = "test11.jpg"
img = Image.open(uri)
results = model.predict(source=img,save=True, project="results",name="result")

