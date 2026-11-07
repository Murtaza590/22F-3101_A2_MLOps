import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_split, make_ds

p, y = load_split("train")
ds = make_ds(p, y, 128, 16, train=True, augment=True)
imgs, labels = next(iter(ds))
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for ax, im, lb in zip(axes.ravel(), imgs.numpy(), labels.numpy()):
    ax.imshow(im.astype("uint8")); ax.axis("off")
    ax.set_title("AI" if lb == 1 else "REAL")
plt.suptitle("One augmented training batch")
plt.tight_layout(); plt.savefig("figures/fig5_augmented_batch.png", dpi=120)