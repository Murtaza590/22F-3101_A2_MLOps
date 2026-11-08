import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

m = tf.keras.models.load_model("api/model.keras")
size = m.input_shape[1]

conv = tf.lite.TFLiteConverter.from_keras_model(m)
open("api/model.tflite", "wb").write(conv.convert())

interp = tf.lite.Interpreter(model_path="api/model.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

paths = pd.read_csv("splits/test.txt")["path"].tolist()[:60]
diffs, agree = [], 0
for p in paths:
    a = np.asarray(Image.open(p).convert("RGB"), dtype="float32")
    pk = float(m.predict(tf.image.resize(a, (size, size))[None], verbose=0)[0][0])
    x = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)[None].astype("float32")
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    pt = float(interp.get_tensor(out["index"])[0][0])
    diffs.append(abs(pk - pt))
    agree += (pk >= 0.5) == (pt >= 0.5)
print("input size:", size)
print("max |keras - tflite| prob difference:", round(max(diffs), 4))
print(f"same label at 0.5: {agree}/{len(paths)}")