# Experiment Tracking — MLflow

> 🚧 **Stub — to be filled in.** Full notes: [roadmap.md §2.2](../../roadmap.md#22-experiment-tracking).
> The open-source industry standard — tracking **plus** a model registry.

## 🎯 Goal
Track YOLO experiments with MLflow and view them in the MLflow UI.

## ⚡ Quick start
```bash
pip install mlflow
yolo settings mlflow=True     # enable the Ultralytics ↔ MLflow integration
# train normally, then view the UI:
mlflow ui                     # opens http://127.0.0.1:5000
```

## 🔗 Links
- 📖 [Roadmap → §2.2](../../roadmap.md#22-experiment-tracking)
- 📚 [Ultralytics × MLflow](https://docs.ultralytics.com/integrations/mlflow)
