# 1.4 Pose / Keypoint Estimation

> 🚧 **Stub — to be filled in.** Full notes & walkthrough:
> [roadmap.md §1.4](../../roadmap.md#14-pose--keypoint-estimation).

## 🎯 Goal
Detect **keypoints** (e.g., human joints) and connect them into a skeleton.

## In this folder
- [`train.py`](train.py) — runnable training template
- `data.yaml` + keypoint annotations — *to be added*

## 🧠 Quick reference
- Label format: box + keypoints `px py visibility` per point (COCO = 17 keypoints)
- Model: `yolo11n-pose.pt` · Demo data: `coco8-pose.yaml`

## 🔗 Links
- 📖 [Roadmap → §1.4](../../roadmap.md#14-pose--keypoint-estimation)
- 📚 [Pose task docs](https://docs.ultralytics.com/tasks/pose/)
