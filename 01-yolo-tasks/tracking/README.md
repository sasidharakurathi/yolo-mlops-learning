# 1.6 Object Tracking & Counting

> 🚧 **Stub — to be filled in.** Full notes & walkthrough:
> [roadmap.md §1.6](../../roadmap.md#16-object-tracking--counting).

## 🎯 Goal
Follow objects **across video frames** with persistent IDs — then count / measure speed.

## In this folder
- [`track.py`](track.py) — runnable tracking template
- A sample video — *to be added*

## 🧠 Quick reference
- Not a separate training task — adds a tracker (**ByteTrack** / **BoT-SORT**) on top of a detector
- `model.track(..., persist=True)` keeps IDs stable between frames

## 🔗 Links
- 📖 [Roadmap → §1.6](../../roadmap.md#16-object-tracking--counting)
- 📚 [Track mode](https://docs.ultralytics.com/modes/track/) · [Solutions (counting)](https://docs.ultralytics.com/solutions/)
