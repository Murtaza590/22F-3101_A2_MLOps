import keras_tuner as kt
import keras

from data import load_split, make_ds

SIZE, BATCH = 128, 32
tr_p, tr_y = load_split("train")
va_p, va_y = load_split("val")
n_tr, n_va = min(1500, len(tr_p)), min(500, len(va_p))
train_ds = make_ds(tr_p[:n_tr], tr_y[:n_tr], SIZE, BATCH, train=True, augment=True)
val_ds = make_ds(va_p[:n_va], va_y[:n_va], SIZE, BATCH)


def build(hp):
    L = keras.layers
    base = keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3),
                                          include_top=False, weights="imagenet")
    n_unfreeze = hp.Choice("unfrozen_layers", [0, 10, 30, 60])
    base.trainable = n_unfreeze > 0
    for l in base.layers[:len(base.layers) - n_unfreeze]:
        l.trainable = False
    for l in base.layers:                       # keep BatchNorm frozen
        if isinstance(l, L.BatchNormalization):
            l.trainable = False
    inp = keras.Input((SIZE, SIZE, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)
    out = L.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-3, 3e-4, 1e-4, 3e-5])),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


tuner = kt.RandomSearch(build, objective="val_loss", max_trials=8,
                        directory="tuner_logs", project_name="visionguard",
                        overwrite=True, seed=42)
tuner.search(train_ds, validation_data=val_ds, epochs=4, verbose=2)

print("\n=== VisionGuard tuner results ===")
print("rank | unfrozen | dropout | lr | val_loss")
rows = []
for i, t in enumerate(tuner.oracle.get_best_trials(8), 1):
    hp = t.hyperparameters.values
    rows.append((i, hp["unfrozen_layers"], hp["dropout"], hp["lr"], t.score))
    print(f"{i} | {hp['unfrozen_layers']} | {hp['dropout']:.1f} | {hp['lr']} | {t.score:.4f}")
print("Best:", tuner.get_best_hyperparameters(1)[0].values)