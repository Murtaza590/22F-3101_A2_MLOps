import argparse
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--best", required=True)
p.add_argument("--second", required=True)
a = p.parse_args()

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
NAME = "VisionGuard"


def model_uri(run_name):
    r = mlflow.search_runs(experiment_names=["VisionGuard"],
                           filter_string=f"tags.mlflow.runName = '{run_name}'",
                           max_results=1)
    rid = r.iloc[0]["run_id"]
    try:
        lms = mlflow.search_logged_models(
            filter_string=f"source_run_id = '{rid}'", output_format="list")
        if lms:
            return f"models:/{lms[0].model_id}"
    except Exception as e:
        print("search_logged_models failed:", e)
    return f"runs:/{rid}/model"


v1 = mlflow.register_model(model_uri(a.second), NAME)
v2 = mlflow.register_model(model_uri(a.best), NAME)
print("Registered versions:", v1.version, v2.version)