from itertools import chain, combinations
from aim.keras import AimCallback
import numpy as np
import keras
import utils
import aim
from keras import layers
import gc
from sklearn.metrics import f1_score
import tensorflow as tf
from datetime import datetime


tf.config.optimizer.set_jit(False)


SEED = 42
IMG_SIZE = (300, 300)
EPOCHS = 1
BATCH_SIZE = 16
FILE_NAME = "out_augmentation"


def v4_model(augmentation):
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(x)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=128, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=256, kernel_size=3,
                      activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v4_model")


(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

train_y_dict = {'lvl1': train_y["lvl1"], 'lvl2': train_y["lvl2"]}
val_y_dict = {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]}
validation_data = (val_x, {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]})

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_lvl1_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

aim_cb = AimCallback(
    repo=".",
    experiment="Vehicle_lvl_Test",
)

sample_weight = {
    "lvl1": np.ones(len(train_y)),
    "lvl2": train_y['lvl1'].to_numpy()
}

data_augmentation = [
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.05, seed=SEED),
    layers.RandomZoom(
        height_factor=(-0.03, 0.01),
        width_factor=(-0.03, 0.01),
        fill_mode="reflect",
        seed=SEED
    ),
    keras.layers.RandomTranslation(
        height_factor=0.03,
        width_factor=0.03,
        fill_mode="reflect",
        seed=SEED
    ),
    layers.RandomContrast(0.2, seed=SEED),
    layers.RandomBrightness(factor=0.2, seed=SEED)
]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


best_loss = 99999
best_combo = ""
power_set = list(powerset(data_augmentation))


with open(FILE_NAME, mode="w") as f:
    for combo in power_set[1:]:
        now = datetime.now()
        print(now, file=f)

        print("combo =", combo, file=f)

        model = v4_model(keras.Sequential([*combo]))

        opt = keras.optimizers.Adam(learning_rate=6e-4)

        model.compile(
            optimizer=opt,
            loss={
                "lvl1": keras.losses.BinaryCrossentropy(),
                "lvl2": keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "lvl1": [keras.metrics.BinaryAccuracy(name="acc")],
            },
            weighted_metrics={
                "lvl2": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            },
        )

        history = model.fit(
            x=train_x,
            y=train_y_dict,
            batch_size=BATCH_SIZE,
            callbacks=[aim_cb],
            sample_weight=sample_weight,
            validation_data=validation_data,
            epochs=EPOCHS
        )

        val = model.evaluate(x=val_x, y=val_y_dict, return_dict=True)
        preds = model.predict(val_x, verbose="0")
        p1_probs = preds['lvl1'].flatten()
        y_pred = (p1_probs >= 0.5).astype(int)
        current_f1 = f1_score(val_y['lvl1'], y_pred)

        val_losses = history.history["val_loss"]
        best_epoch_index = int(np.argmin(val_losses))
        best_epoch = best_epoch_index + 1

        if (best_loss > val_losses[best_epoch_index]):
            best_loss = val_losses[best_epoch_index]
            best_combo = str(combo)

        print(f"Best epoch (1-based): {best_epoch}", file=f)
        print(
            f"val_loss at best epoch: {val_losses[best_epoch_index]}", file=f)

        print(history.history.items(), file=f)

        print(val, file=f)
        print(file=f)
        print(file=f)
        print(file=f)
        print(file=f, flush=True)
        del model
        gc.collect()
    print("best combo: ", best_combo, "  best_loss: ", best_loss, file=f)
