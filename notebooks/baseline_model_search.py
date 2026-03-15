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
EPOCHS = 150
BATCH_SIZE = 32
FILE_NAME = "out_baseline_padding.txt"


def faiga_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="Faiga_model")


def v1_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(filters=32, kernel_size=3,
                      activation="relu", strides=2, padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=3,
                      activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=3,
                      activation="relu", strides=2, padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=3,
                      activation="relu", strides=2, padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v1_model")


def v2_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
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
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v2_model")


def v3_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=4,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=32, kernel_size=4,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=4,
                      activation="relu", padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=4,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=4,
                      activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v3_model")


def v4_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
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
    x = layers.Dense(128, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v4_model")


def v5_model():
    input = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(input)

    # Block 1
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.MaxPool2D()(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.MaxPool2D()(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    x = layers.MaxPool2D()(x)

    # Keep more spatial info (optional: fewer pooling layers if you want even more params)
    x = layers.GlobalAveragePooling2D()(x)

    # Big dense head -> lots of parameters
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="GPT")


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

with open(FILE_NAME, mode="w") as f:
    for create_model in [faiga_model, v2_model, v3_model, v4_model, v5_model]:
        now = datetime.now()
        print(now, file=f)

        model = create_model()
        model.summary(print_fn=lambda x: f.write(x + '\n'))

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

        model.fit(x=train_x, y=train_y_dict, batch_size=BATCH_SIZE, callbacks=[
                  aim_cb], sample_weight=sample_weight, validation_data=validation_data, epochs=EPOCHS)
        val = model.evaluate(x=val_x, y=val_y_dict, return_dict=True)
        preds = model.predict(val_x, verbose="0")
        p1_probs = preds['lvl1'].flatten()
        y_pred = (p1_probs >= 0.5).astype(int)
        current_f1 = f1_score(val_y['lvl1'], y_pred)

        print(f"Resultat F1: {current_f1:.4f}", file=f)
        print(val, file=f)
        print(file=f)
        print(file=f)
        print(file=f)
        print(file=f, flush=True)
        del model
        gc.collect()
