import matplotlib
matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)          # e.g. mnet_finetune
a = p.parse_args()

pred = pd.read_csv(f"outputs/{a.run}/test_predictions.csv")
audit = pd.read_csv("audit.csv")[["path", "width", "height", "file_size"]]
pred = pred.merge(audit, on="path", how="left")
wrong = pred[pred["label"] != pred["pred"]].copy()
print(f"{len(wrong)} misclassified out of {len(pred)}")

# the most confident mistakes are the most informative
wrong["conf_wrong"] = (wrong["prob_ai"] - 0.5).abs()
worst = wrong.sort_values("conf_wrong", ascending=False).head(10)
worst.to_csv(f"outputs/{a.run}/top10_errors.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
    ax.imshow(Image.open(r["path"]).convert("RGB"))
    true = "AI" if r["label"] == 1 else "REAL"
    guess = "AI" if r["pred"] == 1 else "REAL"
    ax.set_title(f"true={true}, pred={guess}\np(ai)={r['prob_ai']:.2f}\n"
                 f"{int(r['width'])}x{int(r['height'])}, {r['file_size']/1024:.0f} KB",
                 fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"figures/fig6_errors_{a.run}.png", dpi=110)
print(worst[["path", "label", "prob_ai", "width", "height", "file_size"]])