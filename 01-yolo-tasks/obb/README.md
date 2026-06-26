# 1.5 Oriented Bounding Boxes (OBB)

> 🚧 **Stub — to be filled in.** Full notes & walkthrough:
> [roadmap.md §1.5](../../roadmap.md#15-oriented-bounding-boxes-obb).

## 🎯 Goal
Draw **rotated** boxes for objects that aren't axis-aligned (aerial imagery, ships, docs).

## In this folder
- [`train.py`](train.py) — runnable training template
- `data.yaml` + rotated-box annotations — *to be added*

## 🧠 Quick reference
- Label format: 4 corners `x1 y1 x2 y2 x3 y3 x4 y4` (normalized)
- Model: `yolo11n-obb.pt` · Demo data: `dota8.yaml`

## 🔗 Links
- 📖 [Roadmap → §1.5](../../roadmap.md#15-oriented-bounding-boxes-obb)
- 📚 [OBB task docs](https://docs.ultralytics.com/tasks/obb/)
