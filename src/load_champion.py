import os
import sys

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
NAME = "VisionGuard"
client = MlflowClient()

mv, how = None, ""
try:
    mv = client.get_model_version_by_alias(NAME, "champion")
    how = "alias"
except Exception:
    pass
if mv is None:
    champs = [v for v in client.search_model_versions(f"name='{NAME}'")
              if v.tags.get("champion") == "true"]
    if champs:
        mv = max(champs, key=lambda v: int(v.version))
        how = "tag"
if mv is None:
    mv = max(client.search_model_versions(f"name='{NAME}'"),
             key=lambda v: int(v.version))
    how = "latest version (no alias or tag found)"

run = client.get_run(mv.run_id)
threshold = run.data.metrics["threshold"]
print(f"Champion found by {how}: version {mv.version}, "
      f"run {run.info.run_name}, threshold {threshold:.2f}")

model = mlflow.keras.load_model(mv.source)
size = model.input_shape[1]

img = tf.io.decode_image(tf.io.read_file(sys.argv[1]), channels=3,
                         expand_animations=False)
img = tf.image.resize(img, (size, size))
prob = float(model.predict(img[None], verbose=0)[0][0])
label = "AI-generated" if prob >= threshold else "Real"
print(f"p(ai) = {prob:.3f} -> {label}")