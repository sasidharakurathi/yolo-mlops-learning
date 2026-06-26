"""
Pose / Keypoint Estimation — training template.
Notes & walkthrough: ../../roadmap.md  (section 1.4)

Run:  python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # pose model

# TODO: swap "coco8-pose.yaml" for your own keypoint-annotated dataset
model.train(data="coco8-pose.yaml", epochs=50, imgsz=640)

# results = model.predict(source=0, show=True)   # live webcam pose
# keypoints = results[0].keypoints
