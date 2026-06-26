"""
Instance Segmentation — training template.
Notes & walkthrough: ../../roadmap.md  (section 1.2)

Run:  python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # segmentation model

# TODO: swap "coco8-seg.yaml" for your own polygon-annotated dataset
model.train(data="coco8-seg.yaml", epochs=50, imgsz=640)

# results = model.predict("https://ultralytics.com/images/bus.jpg", save=True)
# masks = results[0].masks   # the pixel masks
