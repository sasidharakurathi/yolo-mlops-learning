"""
Oriented Bounding Boxes (OBB) — training template.
Notes & walkthrough: ../../roadmap.md  (section 1.5)

Run:  python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")  # oriented-bounding-box model

# TODO: swap "dota8.yaml" for your own rotated-box dataset
model.train(data="dota8.yaml", epochs=50, imgsz=640)

# results = model.predict("path/to/aerial.jpg", save=True)
# obb = results[0].obb   # rotated-box outputs
