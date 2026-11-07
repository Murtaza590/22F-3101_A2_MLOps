import pandas as pd
import tensorflow as tf
import keras

AUTOTUNE = tf.data.AUTOTUNE

augmenter = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
], name="augmentation")


def load_split(name):
    df = pd.read_csv(f"splits/{name}.txt")
    return df["path"].tolist(), df["label"].values.astype("float32")


def make_ds(paths, labels, size, batch, train=False, augment=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (size, size))      # float32, values 0-255
        return img, y

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if train and augment:                             # augmentation: train only
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)