import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
for mv in c.search_model_versions("name='VisionGuard'"):
    print("version", mv.version, "| run", mv.run_id, "| aliases", mv.aliases,
          "| tags", mv.tags, "| source", mv.source)