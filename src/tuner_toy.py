import keras_tuner as kt
import keras

(x, y), _ = keras.datasets.fashion_mnist.load_data()
x = x[:12000] / 255.0
y = y[:12000]
xt, yt, xv, yv = x[:10000], y[:10000], x[10000:], y[10000:]


def build(hp):
    m = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6,
                        directory="tuner_logs", project_name="toy", overwrite=True,
                        seed=42)
tuner.search(xt, yt, validation_data=(xv, yv), epochs=5, verbose=2)

print("\n=== Toy results ===")
print("rank | units | lr | val_accuracy")
for i, t in enumerate(tuner.oracle.get_best_trials(6), 1):
    hp = t.hyperparameters.values
    print(f"{i} | {hp['units']} | {hp['lr']} | {t.score:.4f}")
print("Best:", tuner.get_best_hyperparameters(1)[0].values)