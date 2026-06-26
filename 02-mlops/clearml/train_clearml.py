"""
Phase 2.2 - Your first YOLO training run, tracked with ClearML.

What this script does:
  1. Starts a ClearML "Task" (one tracked experiment).
  2. Trains a tiny YOLO model on the built-in 'coco8' demo dataset (just 8 images).
  3. ClearML automatically logs the settings, metrics, plots, and the trained model
     to your online dashboard - no extra code needed for the logging.

How to run it:
  1. Make sure your virtual environment is active and ClearML is connected
     (see this folder's README for the 'clearml-init' step).
  2. Run:   python train_clearml.py
  3. Open https://app.clear.ml and watch this run appear live.

Beginner note: you do NOT need a GPU. 'coco8' is so small it trains in a minute or two
on a normal laptop CPU.
"""

from clearml import Task
from ultralytics import YOLO

# 1) Start tracking. This creates an experiment ("Task") in your ClearML dashboard.
#    - project_name: a folder that groups related experiments
#    - task_name:    the name of THIS specific run
task = Task.init(
    project_name="YOLO-MLOps-Learning",
    task_name="clearml-first-run",
)

# 2) Choose a model. 'yolo11n.pt' is the smallest/fastest ("n" = nano) and downloads
#    automatically the first time. You can swap in "yolo26n.pt" to match the official
#    Ultralytics x ClearML guide - both work the same way here.
model_variant = "yolo11n.pt"
task.set_parameter("model_variant", model_variant)  # records your choice in ClearML
model = YOLO(model_variant)

# 3) Training settings. 'coco8.yaml' is a tiny 8-image demo dataset that auto-downloads.
#    'epochs' = how many times we pass over the data. Small here so it finishes fast.
args = dict(data="coco8.yaml", epochs=10, imgsz=640)
task.connect(args)  # records these settings in ClearML so the run is reproducible

# 4) Train. Ultralytics' built-in ClearML integration auto-logs metrics, loss curves,
#    sample images, and the final model while this runs.
results = model.train(**args)

print("\nDone! Open https://app.clear.ml and click into 'YOLO-MLOps-Learning' to explore this run.")
