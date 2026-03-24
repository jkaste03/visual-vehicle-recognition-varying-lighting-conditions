#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hierarchical CNN baseline on preprocessed images. With data augmentation."
    )
    parser.add_argument("--run-tag", default="baseline_architecture with augmentation and dropout",
                        help="Checkpoint/output folder name.")
    parser.add_argument("--experiment-name", default=None,
                        help="AIM experiment name. Defaults to run-tag.")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--img-size", type=int, default=300,
                        help="Square image size in pixels.")
    parser.add_argument("--learning-rate", type=float,
                        default=6e-4, help="Adam learning rate.")
    parser.add_argument("--disable-aim", action="store_true",
                        help="Disable AIM logging.")
    return parser.parse_args()


ARGS = parse_args()
SEED = ARGS.seed
IMG_SIZE = (ARGS.img_size, ARGS.img_size)
BATCH_SIZE = ARGS.batch_size
RUN_TAG = ARGS.run_tag
EXPERIMENT_NAME = ARGS.experiment_name or RUN_TAG

tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("RUN_TAG:", RUN_TAG)
print("EXPERIMENT_NAME:", EXPERIMENT_NAME)
print("EPOCHS:", ARGS.epochs)
print("BATCH_SIZE:", BATCH_SIZE)


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "datasplitt_preprocessed").exists():
            return p
        if (p / "datasplitt" / "train.csv").exists():
            return p
    raise FileNotFoundError(
        "Fant ikke prosjektrot med datasplitt_preprocessed/ eller datasplitt/train.csv")


PROJECT_ROOT = find_project_root(Path.cwd())
TRAIN_CSV = PROJECT_ROOT / "datasplitt_preprocessed" / "train_processed.csv"
VAL_CSV = PROJECT_ROOT / "datasplitt_preprocessed" / "val_processed.csv"
TEST_CSV = PROJECT_ROOT / "datasplitt_preprocessed" / "test_processed.csv"
IMG_ROOT = PROJECT_ROOT / "datasett_preprocessed"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("VAL_CSV:", VAL_CSV)
print("TEST_CSV:", TEST_CSV)
print("IMG_ROOT:", IMG_ROOT)

if not TRAIN_CSV.exists() or not VAL_CSV.exists() or not TEST_CSV.exists():
    raise FileNotFoundError(
        "Fant ikke de preprosesserte CSV-filene. Kjør baseline_preprocessing.ipynb først.")

if not IMG_ROOT.exists():
    raise FileNotFoundError(f"Finner ikke IMG_ROOT: {IMG_ROOT}")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"[{name}] shape={df.shape}")
    print(f"[{name}] columns={df.columns.tolist()}")

PATH_COL = "processed_image" if "processed_image" in train_df.columns else "image"
print("Bruker bildefilsti-kolonne:", PATH_COL)

required_cols = [
    PATH_COL,
    "lvl1",
    "lvl2",
    "lighting",
    "y_lvl1",
    "y_lvl2",
    "w_lvl2",
]

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name}_df mangler forventede kolonner {missing}. "
            "Kjør baseline_preprocessing.ipynb på nytt og eksporter oppdaterte CSV-er."
        )

for df in (train_df, val_df, test_df):
    if "w_lvl1" not in df.columns:
        df["w_lvl1"] = 1.0

sample_paths = [str(IMG_ROOT / p)
                for p in train_df[PATH_COL].astype(str).head(5)]
for p in sample_paths:
    print(Path(p).exists(), p)

num_lvl2 = int(train_df.loc[train_df["w_lvl2"] > 0, "y_lvl2"].max()) + 1
lvl2_classes = (
    train_df.loc[train_df["w_lvl2"] > 0, ["lvl2", "y_lvl2"]]
    .drop_duplicates()
    .sort_values("y_lvl2")["lvl2"]
    .tolist()
)

print("num_lvl2:", num_lvl2)
print("lvl2_classes:", lvl2_classes)


def decode_ready_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


def apply_conditional_brightness(img: tf.Tensor, lighting: tf.Tensor) -> tf.Tensor:
    """
    Bruk brightness bare for bilder merket som Medium eller Dark.
    Faktor 0.08 er et forsiktig startpunkt.
    """
    lighting = tf.strings.lower(lighting)

    def brighten():
        x = tf.image.random_brightness(img, max_delta=0.12)
        return tf.clip_by_value(x, 0.0, 1.0)

    def unchanged():
        return img

    return tf.cond(
        tf.logical_or(
            tf.equal(lighting, "medium"),
            tf.equal(lighting, "dark"),
        ),
        brighten,
        unchanged,
    )


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = np.array([str(IMG_ROOT / p)
                     for p in df[PATH_COL].astype(str).to_list()], dtype=np.str_)
    y1 = df["y_lvl1"].to_numpy(np.int32)
    y2 = df["y_lvl2"].to_numpy(np.int32)
    w1 = df["w_lvl1"].to_numpy(np.float32)
    w2 = df["w_lvl2"].to_numpy(np.float32)
    lighting = df["lighting"].astype(str).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, y1, y2, w1, w2, lighting))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map(path, y1, y2, w1, w2, lighting):
        img = decode_ready_image(path)

        if training:
            img = apply_conditional_brightness(img, lighting)

        y = {"lvl1": y1, "lvl2": y2}
        sw = {"lvl1": w1, "lvl2": w2}
        return img, y, sw

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

    if not training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)

for x, y, sw in train_ds.take(1):
    print("x:", x.shape, x.dtype)
    print("y lvl1:", y["lvl1"].shape, "y lvl2:", y["lvl2"].shape)
    print("sw lvl1:", sw["lvl1"].shape, "sw lvl2:", sw["lvl2"].shape)


def build_augmentation():
    return keras.Sequential([
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
    ], name="best_data_augmentation")


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = build_augmentation()(inputs)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.45)(x)

    out_lvl1 = keras.layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    out_lvl2 = keras.layers.Dense(
        num_lvl2, activation="softmax", name="lvl2")(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"lvl1": out_lvl1, "lvl2": out_lvl2},
        name="Modell_med_augmentering_dropout",
    )

    opt = keras.optimizers.Adam(learning_rate=ARGS.learning_rate)

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
        jit_compile=False,
    )
    return model


model = build_model()
model.summary()

callbacks: list[keras.callbacks.Callback] = []

if not ARGS.disable_aim:
    from aim.tensorflow import AimCallback

    aim_callback = AimCallback(
        repo=str(PROJECT_ROOT),
        experiment=EXPERIMENT_NAME,
        log_system_params=False,
    )
    callbacks.append(aim_callback)
    print("Aim repo:", PROJECT_ROOT / ".aim")
    print("Aim experiment:", EXPERIMENT_NAME)
else:
    print("AIM logging disabled.")

ckpt_dir = PROJECT_ROOT / "checkpoints" / RUN_TAG
ckpt_dir.mkdir(parents=True, exist_ok=True)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=str(ckpt_dir / "epoch_{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_best_only=False,
    verbose=0,
)
callbacks.append(checkpoint_callback)

print("Checkpoint dir:", ckpt_dir)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=ARGS.epochs,
    callbacks=callbacks,
)

print("Eval (val; lvl2 maskert via sample_weight) – siste epoch, ikke beste:")
val_metrics = model.evaluate(val_ds, return_dict=True)
print(val_metrics)

history_dir = PROJECT_ROOT / "results" / RUN_TAG / "train"
history_dir.mkdir(parents=True, exist_ok=True)

history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
history_path = history_dir / "history.csv"
history_df.to_csv(history_path, index=False, encoding="utf-8")

meta = {
    "run_tag": RUN_TAG,
    "experiment_name": EXPERIMENT_NAME,
    "epochs": ARGS.epochs,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "img_size": IMG_SIZE,
    "learning_rate": ARGS.learning_rate,
    "num_lvl2": num_lvl2,
    "val_metrics_last_epoch": {k: float(v) for k, v in val_metrics.items()},
}
with open(history_dir / "train_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Lagret history:", history_path)
print("Lagret metadata:", history_dir / "train_meta.json")
print("Ferdig. Kjør evaluering i notebooken etterpå.")
