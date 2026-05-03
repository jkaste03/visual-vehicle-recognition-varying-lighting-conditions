from tensorflow import keras
from keras import layers, regularizers
from aim.keras import AimCallback  # remove if you don't use it
import numpy as np
import utils
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
BATCH_SIZE = 64
FILE_NAME = "out_bayesian_search_dropout_l2.txt"
MAX_TRIALS = 30

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

train_y_dict = {"lvl1": train_y["lvl1"], "lvl2": train_y["lvl2"]}
val_y_dict = {"lvl1": val_y["lvl1"], "lvl2": val_y["lvl2"]}
validation_data = (val_x, {"lvl1": val_y["lvl1"], "lvl2": val_y["lvl2"]})

sample_weight = {
    "lvl1": np.ones(len(train_y)),
    "lvl2": train_y["lvl1"].to_numpy(),
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
# Fixed augmentation layer (your "best_data_augmentation")
# -------------------------------------------------------

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", name="rand_flip"),
        layers.RandomZoom(
            height_factor=(-0.10, 0.03),
            width_factor=(-0.10, 0.03),
            fill_mode="reflect",
            name="rand_zoom",
        ),
        layers.RandomContrast(
            factor=0.10,
            name="rand_contrast",
        ),
    ],
    name="best_data_augmentation",
)

# -------------------------------------------------------
# Base model builder using given augmentation + hp for dropout & l2
# -------------------------------------------------------


def v4_model(augmentation, l2_factor=0.01, dropout_rate=0.5):
    # l2_factor = 0.0  → no L2
    # dropout_rate = 0.0 → no Dropout

    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)

    reg = regularizers.l2(l2_factor) if l2_factor > 0.0 else None

    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(
        filters=256,
        kernel_size=3,
        activation="relu",
        kernel_regularizer=reg,
    )(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    model = keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2},
                        name="v4_model")

    # >>> IMPORTANT: compile the model here <<<
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
# HyperModel / build function for Keras Tuner (dropout & l2)
# -------------------------------------------------------


def build_model(hp: kt.HyperParameters):
    # Dropout rate: 0.0–0.5
    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        default=0.0,
    )

    # Allow "no L2" + positive L2 values (log-sampled)
    use_l2 = hp.Boolean("use_l2", default=True)
    if use_l2:
        l2_factor = hp.Float(
            "l2_factor",
            min_value=1e-7,
            max_value=1e-3,
            sampling="log",
            default=1e-4,
        )
    else:
        l2_factor = 0.0

    model = v4_model(
        augmentation=augmentation,
        l2_factor=l2_factor,
        dropout_rate=dropout_rate,
    )
    return model


# -------------------------------------------------------
# Bayesian Optimization Tuner
# -------------------------------------------------------
tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("val_score_50_50", direction="max"),
    max_trials=MAX_TRIALS,
    num_initial_points=8,
    directory="bayesian_search",
    project_name="vehicle_lvl_dropout_l2",
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

    # Predictions
    preds = best_model.predict(val_x, verbose=0)
    p1_probs = preds["lvl1"].flatten()      # shape (N,)
    p2_probs = preds["lvl2"]                # shape (N, 7)

    # lvl1 F1
    y1_pred = (p1_probs >= 0.5).astype(int)
    f1_lvl1 = f1_score(val_y["lvl1"], y1_pred)

    # lvl2 F1 (macro)
    y2_pred = p2_probs.argmax(axis=1)
    f1_lvl2_macro = f1_score(val_y["lvl2"], y2_pred, average="macro")

    # Combined 50/50 score
    score_50_50 = 0.5 * f1_lvl1 + 0.5 * f1_lvl2_macro

    print("Validation metrics (Keras):", file=f)
    print(val, file=f)

    print("Custom F1 metrics:", file=f)
    print(f"F1 (lvl1): {f1_lvl1}", file=f)
    print(f"F1 (lvl2 macro): {f1_lvl2_macro}", file=f)
    print(f"Combined 50/50 score: {score_50_50}", file=f)

    print(file=f, flush=True)

gc.collect
