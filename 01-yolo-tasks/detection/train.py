"""
Object Detection — training template.
Notes & walkthrough: ../../roadmap.md  (section 1.1)

Run:  python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pretrained detection model ("n" = nano = smallest/fastest)

# TODO: swap "coco8.yaml" for your own data.yaml once you have a custom dataset
model.train(data="coco8.yaml", epochs=50, imgsz=640, batch=16)

model.val()  # evaluate → prints mAP, precision, recall

# Try these next:
# model.predict("https://ultralytics.com/images/bus.jpg", save=True)  # inference
# model.predict(source=0, show=True)   # live webcam
# model.export(format="onnx")          # export for deployment
