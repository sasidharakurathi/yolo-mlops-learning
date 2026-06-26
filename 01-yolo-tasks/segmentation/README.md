# 1.2 Instance Segmentation

> 🚧 **Stub — to be filled in.** Full notes & walkthrough:
> [roadmap.md §1.2](../../roadmap.md#12-instance-segmentation).

## 🎯 Goal
Outline objects at the **pixel** level (masks), not just boxes.

## In this folder
- [`train.py`](train.py) — runnable training template
- `data.yaml` + sample images/polygons — *to be added*

## 🧠 Quick reference
- Label format: `class_id x1 y1 x2 y2 … xn yn` (polygon, normalized)
- Model: `yolo11n-seg.pt` · Demo data: `coco8-seg.yaml`

## 🔗 Links
- 📖 [Roadmap → §1.2](../../roadmap.md#12-instance-segmentation)
- 📚 [Segment task docs](https://docs.ultralytics.com/tasks/segment/)
