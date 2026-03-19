from aim.keras import AimCallback
import numpy as np
import keras
import utils
from keras import layers
import gc
from sklearn.metrics import f1_score
import tensorflow as tf
from datetime import datetime
import keras_tuner as kt

tf.config.optimizer.set_jit(False)

# ---------- Custom callback: validation F1 for lvl1, lvl2, and combined ----------


class ValF1Callback(keras.callbacks.Callback):
    def __init__(
        self,
        val_x,
        val_y_lvl1,
        val_y_lvl2,
        name_lvl1="val_f1_lvl1",
        name_lvl2="val_f1_lvl2_macro",
        name_combined="val_score_50_50",
        threshold=0.5,
    ):
        super().__init__()
        self.val_x = val_x
        self.val_y_lvl1 = val_y_lvl1
        self.val_y_lvl2 = val_y_lvl2
        self.name_lvl1 = name_lvl1
        self.name_lvl2 = name_lvl2
        self.name_combined = name_combined
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        preds = self.model.predict(self.val_x, verbose=0)
        p1_probs = preds["lvl1"].flatten()
        p2_probs = preds["lvl2"]  # shape (N, 7)

        # lvl1: binary F1 with threshold
        y1_pred = (p1_probs >= self.threshold).astype(int)
        f1_lvl1 = f1_score(self.val_y_lvl1, y1_pred)

        # lvl2: macro F1 over 7 classes
        y2_pred = p2_probs.argmax(axis=1)
        f1_lvl2_macro = f1_score(self.val_y_lvl2, y2_pred, average="macro")

        # combined 50/50 score
        score_50_50 = 0.5 * f1_lvl1 + 0.5 * f1_lvl2_macro

        logs[self.name_lvl1] = f1_lvl1
        logs[self.name_lvl2] = f1_lvl2_macro
        logs[self.name_combined] = score_50_50

        print(
            f"\nEpoch {epoch + 1}: "
            f"{self.name_lvl1} = {f1_lvl1:.4f}, "
            f"{self.name_lvl2} = {f1_lvl2_macro:.4f}, "
            f"{self.name_combined} = {score_50_50:.4f}"
        )
# ------------------------------------------------------------


SEED = 42
IMG_SIZE = (300, 300)
EPOCHS = 200
BATCH_SIZE = 16
FILE_NAME = "out_bayesian_search_2.txt"

# -------------------------------------------------------
# Data loading (unchanged)
# -------------------------------------------------------
(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

train_y_dict = {'lvl1': train_y["lvl1"], 'lvl2': train_y["lvl2"]}
val_y_dict = {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]}
validation_data = (val_x, {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]})

sample_weight = {
    "lvl1": np.ones(len(train_y)),
    "lvl2": train_y['lvl1'].to_numpy()
}

# ----- custom F1 callback + EarlyStopping on that metric -----
val_f1_cb = ValF1Callback(
    val_x=val_x,
    val_y_lvl1=val_y["lvl1"],
    val_y_lvl2=val_y["lvl2"],
    name_lvl1="val_f1_lvl1",
    name_lvl2="val_f1_lvl2_macro",
    name_combined="val_score_50_50",
    threshold=0.5,
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_score_50_50",  # combined 50/50 metric
    mode="max",
    patience=10,
    restore_best_weights=True,
)

# -------------------------------------------------------


# -------------------------------------------------------
# Base model builder (without augmentation inline)
# -------------------------------------------------------


def v4_model(augmentation_layer):
    inputs = x = keras.Input(shape=(*IMG_SIZE, 3))

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Rescaling(1.0 / 255)(x)
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

    model = keras.Model(inputs=inputs, outputs={"lvl1": lvl1, "lvl2": lvl2},
                        name="v4_model")

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
    return model

# -------------------------------------------------------
# HyperModel / build function for Keras Tuner
# -------------------------------------------------------


def build_model(hp: kt.HyperParameters):
    """
    This function defines the search space:
      - Whether each augmentation is used or not.
      - Magnitudes of some augmentations.
    """

    # --- Augmentation hyperparameters and conditional inclusion ---

    # RandomFlip horizontal
    use_flip = hp.Boolean("use_flip", default=True)

    # RandomRotation
    use_rotation = hp.Boolean("use_rotation", default=True)
    max_rotation = hp.Float(
        "max_rotation",
        min_value=0.0,
        max_value=0.15,
        step=0.01,
        default=0.05,
    )

    # RandomZoom
    use_zoom = hp.Boolean("use_zoom", default=True)
    max_zoom_in = hp.Float(
        "max_zoom_in",
        min_value=0.0,
        max_value=0.1,
        step=0.01,
        default=0.03,  # matches your ~0.03 in height/width
    )
    max_zoom_out = hp.Float(
        "max_zoom_out",
        min_value=0.0,
        max_value=0.1,
        step=0.01,
        default=0.01,
    )

    # RandomTranslation
    use_translation = hp.Boolean("use_translation", default=True)
    max_translation = hp.Float(
        "max_translation",
        min_value=0.0,
        max_value=0.1,
        step=0.01,
        default=0.03,
    )

    # RandomContrast
    use_contrast = hp.Boolean("use_contrast", default=True)
    contrast_factor = hp.Float(
        "contrast_factor",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        default=0.2,
    )

    # RandomBrightness (keras >= 2.11)
    use_brightness = hp.Boolean("use_brightness", default=True)
    brightness_factor = hp.Float(
        "brightness_factor",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        default=0.2,
    )

    aug_layers = []

    if use_flip:
        aug_layers.append(
            layers.RandomFlip("horizontal", seed=SEED)
        )

    if use_rotation and max_rotation > 0.0:
        aug_layers.append(
            layers.RandomRotation(max_rotation, seed=SEED)
        )

    if use_zoom and (max_zoom_in > 0.0 or max_zoom_out > 0.0):
        aug_layers.append(
            layers.RandomZoom(
                height_factor=(-max_zoom_in, max_zoom_out),
                width_factor=(-max_zoom_in, max_zoom_out),
                fill_mode="reflect",
                seed=SEED
            )
        )

    if use_translation and max_translation > 0.0:
        aug_layers.append(
            layers.RandomTranslation(
                height_factor=max_translation,
                width_factor=max_translation,
                fill_mode="reflect",
                seed=SEED
            )
        )

    if use_contrast and contrast_factor > 0.0:
        aug_layers.append(
            layers.RandomContrast(contrast_factor, seed=SEED)
        )

    if use_brightness and brightness_factor > 0.0:
        aug_layers.append(
            layers.RandomBrightness(factor=brightness_factor, seed=SEED)
        )

    if len(aug_layers) > 0:
        augmentation_layer = keras.Sequential(
            aug_layers, name="data_augmentation")
    else:
        augmentation_layer = None

    model = v4_model(augmentation_layer)
    return model


# -------------------------------------------------------
# Bayesian Optimization Tuner
# -------------------------------------------------------
tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_score_50_50", direction="max"),
    max_trials=30,               # number of different augmentation configs to try
    num_initial_points=8,        # random initial points
    directory="bayesian_search",
    project_name="vehicle_lvl_aug_2nd",
    overwrite=True,
)

# -------------------------------------------------------
# Run search
# -------------------------------------------------------
tuner.search(
    x=train_x,
    y=train_y_dict,
    sample_weight=sample_weight,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[val_f1_cb, early_stop],
    verbose=1,
)

# -------------------------------------------------------
# Retrieve best model & log info
# -------------------------------------------------------
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

with open(FILE_NAME, mode="w") as f:
    now = datetime.now()
    print(now, file=f)
    print("Best hyperparameters:", file=f)
    for k, v in best_hp.values.items():
        print(f"{k}: {v}", file=f)

    # Evaluate on validation data
    val = best_model.evaluate(x=val_x, y=val_y_dict, return_dict=True)
    preds = best_model.predict(val_x, verbose=0)
    p1_probs = preds['lvl1'].flatten()
    y_pred = (p1_probs >= 0.5).astype(int)
    current_f1 = f1_score(val_y['lvl1'], y_pred)

    print("Validation metrics:", file=f)
    print(val, file=f)
    print(f"F1 (lvl1): {current_f1}", file=f)

    print(file=f, flush=True)

gc.collect()
