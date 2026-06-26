"""
Object Tracking & Counting — template.
Notes & walkthrough: ../../roadmap.md  (section 1.6)

Run:  python track.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # any detection/seg/pose model works

# Track on a video (or source=0 for webcam). persist=True keeps stable IDs across frames.
# TODO: point "source" at your own video file
model.track(source="people.mp4", tracker="bytetrack.yaml", persist=True, save=True)

# For counting objects crossing a line/region, see Ultralytics solutions.ObjectCounter:
# https://docs.ultralytics.com/solutions/
