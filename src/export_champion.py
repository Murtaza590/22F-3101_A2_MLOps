import json
import os
import shutil
from pathlib import Path

import mlflow
from dotenv import load_dotenv

RUN = "mnet_finetune"          # change if your best run has another name
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
r = mlflow.search_runs(experiment_names=["VisionGuard"],
                       filter_string=f"tags.mlflow.runName = '{RUN}'",
                       max_results=1).iloc[0]
thr = float(r["metrics.threshold"])
Path("api").mkdir(exist_ok=True)
shutil.copy(f"models/{RUN}.keras", "api/model.keras")
json.dump({"run": RUN, "run_id": r["run_id"], "threshold": thr,
           "model_version": "VisionGuard v2 (champion)"},
          open("api/model_info.json", "w"), indent=2)
print("Exported", RUN, "threshold", thr)