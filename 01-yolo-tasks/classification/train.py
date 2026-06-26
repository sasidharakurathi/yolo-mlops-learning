"""
Image Classification — training template.
Notes & walkthrough: ../../roadmap.md  (section 1.3)

Run:  python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # classification model

# TODO: swap "mnist160" for a folder dataset organized as train/<class>/ + val/<class>/
model.train(data="mnist160", epochs=20, imgsz=64)

# result = model.predict("path/to/image.jpg")
# print(result[0].probs.top1)   # predicted class index
