# 2.6 Deployment & Serving

> 🚧 **Stub — to be filled in.** Full notes: [roadmap.md §2.6](../../roadmap.md#26-deployment--serving).

## 🎯 Goal
Get your model out of the notebook: export it, wrap it in an API, containerize it.

## ⚡ Quick start
```bash
yolo export model=best.pt format=onnx      # → best.onnx (portable)
```
Then serve with **FastAPI** + **Docker** (full example to be added here):
`app.py` (FastAPI `/predict`) → `Dockerfile` → `docker run`.

## What will live here
- `export_demo.py` · `app.py` (FastAPI) · `Dockerfile`

## 🔗 Links
- 📖 [Roadmap → §2.6](../../roadmap.md#26-deployment--serving)
- 📚 [Export mode](https://docs.ultralytics.com/modes/export/) · [Deployment options](https://docs.ultralytics.com/guides/model-deployment-options/)
