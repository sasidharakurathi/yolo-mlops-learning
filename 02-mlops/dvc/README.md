# 2.3 Data & Model Versioning — DVC

> 🚧 **Stub — to be filled in.** Full notes: [roadmap.md §2.3](../../roadmap.md#23-data--model-versioning).

## 🎯 Goal
Version datasets and model weights like code — without bloating Git.

## ⚡ Quick start
```bash
pip install dvc
dvc init
dvc add data/my_dataset                 # creates a small .dvc pointer file
git add data/my_dataset.dvc .gitignore && git commit -m "track dataset v1"
dvc remote add -d storage gdrive://<folder-id>
dvc push                                # uploads data to the remote
```

## 🔗 Links
- 📖 [Roadmap → §2.3](../../roadmap.md#23-data--model-versioning)
- 📚 [Ultralytics × DVC](https://docs.ultralytics.com/integrations/dvc) · [DVC Get Started](https://dvc.org/doc/start)
