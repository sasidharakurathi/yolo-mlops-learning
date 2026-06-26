# 2.4 Hyperparameter Optimization (HPO)

> 🚧 **Stub — to be filled in.** Full notes: [roadmap.md §2.4](../../roadmap.md#24-hyperparameter-optimization).

## 🎯 Goal
Let the machine search for better hyperparameters than you'd pick by hand.

## ⚡ Quick start — Ultralytics built-in tuner
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.tune(data="coco8.yaml", epochs=30, iterations=10, optimizer="AdamW")
```
Scale up later with **Ray Tune** or **ClearML HPO**.

## 🔗 Links
- 📖 [Roadmap → §2.4](../../roadmap.md#24-hyperparameter-optimization)
- 📚 [Tuning guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/) · [× Ray Tune](https://docs.ultralytics.com/integrations/ray-tune)
