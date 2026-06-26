# 2.7 CI/CD & Monitoring

> 🚧 **Stub — to be filled in.** Full notes: [roadmap.md §2.7](../../roadmap.md#27-cicd--monitoring).

## 🎯 Goal
Automate validation on every push, and watch the model for drift in production.

## ⚡ Quick start — GitHub Actions (sketch)
```yaml
# .github/workflows/validate.yml
on: [push]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ultralytics
      - run: python scripts/check_map.py   # fail the build if mAP < threshold
```
Monitoring later: **Evidently AI** for data/concept drift.

## 🔗 Links
- 📖 [Roadmap → §2.7](../../roadmap.md#27-cicd--monitoring)
- 📚 [GitHub Actions](https://docs.github.com/en/actions) · [Evidently AI](https://www.evidentlyai.com/)
