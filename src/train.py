import matplotlib
matplotlib.use("Agg")
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
import keras

from data import load_split, make_ds

p = argparse.ArgumentParser()
p.add_argument("--model", default="cnn", choices=["cnn", "mobilenet"])
p.add_argument("--run_name", required=True)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr_finetune", type=float, default=1e-5)
p.add_argument("--img_size", type=int, default=128)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--ft_epochs", type=int, default=10)
p.add_argument("--augment", type=int, default=1)
p.add_argument("--unfreeze", type=int, default=0)   # fine-tune stage for mobilenet
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--log", default="manual", choices=["manual", "auto"])
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

keras.utils.set_random_seed(a.seed)
load_dotenv()
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("VisionGuard")

out = Path("outputs") / a.run_name
out.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "unknown"


def dvc_version():
    try:
        for line in open("data.dvc"):
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    except Exception:
        pass
    return "unknown"


# ---------- models ----------
def build_cnn(size, dropout):
    L = keras.layers
    inp = keras.Input((size, size, 3))
    x = L.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128):
        x = L.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out_ = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out_, name="baseline_cnn"), None


def build_mobilenet(size, dropout):
    L = keras.layers
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((size, size, 3))
    # same maths as mobilenet_v2.preprocess_input: scale 0-255 to [-1, 1]
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out_ = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out_, name="mobilenetv2_vg"), base


def compile_model(m, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])


class MLflowLogger(keras.callbacks.Callback):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()},
                           step=self.offset + epoch)


def fit_stage(m, lr, epochs, ckpt, offset=0):
    compile_model(m, lr)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-7),
    ]
    if a.log == "manual":
        cbs.append(MLflowLogger(offset))
    h = m.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cbs, verbose=2)
    return h.history


# ---------- data ----------
tr_p, tr_y = load_split("train")
va_p, va_y = load_split("val")
te_p, te_y = load_split("test")
train_ds = make_ds(tr_p, tr_y, a.img_size, a.batch, train=True,
                   augment=bool(a.augment), seed=a.seed)
val_ds = make_ds(va_p, va_y, a.img_size, a.batch)
test_ds = make_ds(te_p, te_y, a.img_size, a.batch)

model, base = (build_cnn if a.model == "cnn" else build_mobilenet)(a.img_size, a.dropout)

with mlflow.start_run(run_name=a.run_name):
    mlflow.set_tags({
        "git_commit": sh("git rev-parse HEAD"),
        "dvc_data_version": dvc_version(),
        "seed": str(a.seed),
        "logging_mode": a.log,
        "label_convention": "ai=1, real=0",
    })
    mlflow.log_params(vars(a))
    if a.log == "auto":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)

    model.summary()
    with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))
    mlflow.log_param("total_params", model.count_params())

    # ---- stage 1 ----
    ck1 = str(out / "stage1.keras")
    h1 = fit_stage(model, a.lr, a.epochs, ck1)
    print("Stage 1 trainable layers:",
          [l.name for l in model.layers if l.trainable])
    hist = {k: list(v) for k, v in h1.items()}
    n1 = len(h1["val_loss"])
    ck_final, stage2_start = ck1, None

    # ---- stage 2: fine-tune last block ----
    if a.model == "mobilenet" and a.unfreeze:
        base.trainable = True
        for l in base.layers:
            l.trainable = (l.name.startswith(("block_16", "Conv_1"))
                           and not isinstance(l, keras.layers.BatchNormalization))
        tl = [l.name for l in base.layers if l.trainable]
        print("Stage 2 trainable layers:", tl)
        (out / "trainable_layers_stage2.txt").write_text("\n".join(tl))
        ck2 = str(out / "stage2.keras")
        h2 = fit_stage(model, a.lr_finetune, a.ft_epochs, ck2, offset=n1)
        for k, v in h2.items():
            hist.setdefault(k, []).extend(v)
        stage2_start = n1

    best_epoch = int(np.argmin(hist["val_loss"]))
    if stage2_start is not None and best_epoch >= stage2_start:
        ck_final = ck2
    best = keras.models.load_model(ck_final)
    mlflow.log_metric("best_epoch", best_epoch + 1)
    mlflow.log_metric("best_val_loss", float(min(hist["val_loss"])))
    print(f"Best epoch: {best_epoch + 1}")

    # ---- training curves with best epoch marked ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (k, t) in enumerate([("loss", "Loss"), ("accuracy", "Accuracy")]):
        ax[i].plot(hist[k], label="train")
        ax[i].plot(hist["val_" + k], label="val")
        ax[i].axvline(best_epoch, color="green", ls="--", label=f"best epoch {best_epoch + 1}")
        if stage2_start:
            ax[i].axvline(stage2_start, color="gray", ls=":", label="fine-tune starts")
        ax[i].set_xlabel("epoch"); ax[i].set_title(t); ax[i].legend()
    plt.tight_layout(); plt.savefig(out / "curves.png", dpi=120); plt.close()

    # ---- threshold from VALIDATION data ----
    val_prob = best.predict(val_ds, verbose=0).ravel()
    ths = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(va_y, val_prob >= t) for t in ths]
    thr = float(ths[int(np.argmax(f1s))])
    mlflow.log_metric("threshold", thr)
    print(f"Threshold chosen on validation (max F1): {thr:.2f}")

    # ---- test set, evaluated once ----
    te_prob = best.predict(test_ds, verbose=0).ravel()
    te_pred = (te_prob >= thr).astype(int)
    m = {
        "test_accuracy": accuracy_score(te_y, te_pred),
        "test_precision": precision_score(te_y, te_pred),
        "test_recall": recall_score(te_y, te_pred),
        "test_f1": f1_score(te_y, te_pred),
        "test_roc_auc": roc_auc_score(te_y, te_prob),
    }
    mlflow.log_metrics(m)
    print({k: round(v, 4) for k, v in m.items()})

    cm = confusion_matrix(te_y, te_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["real", "ai"]); ax.set_yticks([0, 1], ["real", "ai"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()

    fpr, tpr, _ = roc_curve(te_y, te_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {m['test_roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve (test)"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=120); plt.close()

    pd.DataFrame({"path": te_p, "label": te_y, "prob_ai": te_prob,
                  "pred": te_pred}).to_csv(out / "test_predictions.csv", index=False)

    # ---- artifacts ----
    mlflow.log_artifacts(str(out))
    mlflow.log_artifacts("splits", artifact_path="splits")
    sh("pip freeze > requirements_run.txt")
    mlflow.log_artifact("requirements_run.txt")
    shutil.copy(ck_final, f"models/{a.run_name}.keras")
    try:
        mlflow.keras.log_model(best, name="model")
    except TypeError:
        mlflow.keras.log_model(best, artifact_path="model")
    print("Done:", a.run_name)