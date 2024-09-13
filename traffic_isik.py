from ultralytics import YOLO
from PIL import Image
model=YOLO("best.pt")
im1=Image.open("yellow_80.jpg")
sonuc=model.predict(source=im1,save=True)
print(sonuc)

