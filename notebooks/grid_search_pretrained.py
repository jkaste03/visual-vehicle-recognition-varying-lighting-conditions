#!/usr/bin/env python3
import argparse
import datetime
import pathlib

from tensorflow import keras
from tensorflow.keras import layers

import utils  # your module that contains read_andre_data_preprocessed

# ---- Configurable defaults ----
IMG_SIZE = (300, 300)
EPOCHS = 30
num_lvl2 = 7


def build_model(
    img_size=IMG_SIZE,
    num_lvl2=num_lvl2,
    lr=3e-4,
    num_unfrozen_layers=0,
    use_dense=False,
) -> keras.Model:
    inputs = keras.Input(shape=(*img_size, 3), name="image")

    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        input_tensor=inputs,
        pooling="avg",
    )

    # Freeze all layers first
    for l in backbone.layers:
        l.trainable = False

    # Unfreeze the last num_unfrozen_layers layers (top K layers)
    num_unfrozen = min(num_unfrozen_layers, len(backbone.layers))
    if num_unfrozen > 0:
        for l in backbone.layers[-num_unfrozen:]:
            l.trainable = True

    x = backbone.output

    if use_dense:
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

    out_lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    out_lvl2 = layers.Dense(num_lvl2, activation="softmax", name="lvl2")(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"lvl1": out_lvl1, "lvl2": out_lvl2},
        name="efficientnetv2s_hierarchical",
    )

    opt = keras.optimizers.Adam(learning_rate=lr)

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run an EfficientNetV2S experiment with specified top-unfrozen layers and optional dense head."
    )
    p.add_argument(
        "--num-unfrozen-layers",
        type=int,
        default=0,
        help="Number of backbone top layers (last K) to unfreeze (0 => keep all frozen).",
    )
    p.add_argument(
        "--use-dense",
        action="store_true",
        help="Add extra Dense(256) + Dropout before heads if set.",
    )
    p.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to results file to append to. If omitted, a timestamped file is used.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    num_unfrozen_layers = args.num_unfrozen_layers
    use_dense = args.use_dense
    epochs = args.epochs

    # determine results file
    if args.results_file:
        results_path = pathlib.Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        first_run = not results_path.exists()
    else:
        RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = pathlib.Path(f"grid_search_results_{RUN_ID}.txt")
        first_run = True

    # Header if new file
    with open(results_path, "a") as f:
        if first_run:
            header = (
                f"Experiment run: {datetime.datetime.now().isoformat()}\n"
                f"IMG_SIZE={IMG_SIZE}, default_num_lvl2={num_lvl2}\n"
                + "-" * 80
                + "\n"
            )
            print(header, file=f)

    # Print to stdout also
    print(
        f"Running experiment: num_unfrozen_layers={num_unfrozen_layers}, use_dense={use_dense}, epochs={epochs}"
    )
    print(f"Results will be appended to: {results_path}")

    # Load datasets (inside the process)
    train_ds, val_ds, test_ds = utils.read_andre_data_preprocessed()

    # Build model (pass the number of layers to unfreeze)
    model = build_model(
        num_unfrozen_layers=num_unfrozen_layers,
        use_dense=use_dense,
    )

    # Debug: report how many backbone layers were made trainable
    backbone_layers = model.get_layer(
        index=1) if len(model.layers) > 1 else None
    # (Note: getting the backbone by index may vary; simpler: count trainable weights)
    num_trainable = len([w for w in model.trainable_weights])
    print(f"Model built. Trainable weights count: {num_trainable}")

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[],  # no callbacks to reduce retained references
        verbose=1,
    )

    # Evaluate
    val_results = model.evaluate(val_ds, return_dict=True, verbose=0)

    # Compute combined score
    val_lvl1_acc = val_results.get("lvl1_acc", 0.0)
    val_lvl2_acc = val_results.get("lvl2_acc", 0.0)
    score = (val_lvl1_acc + val_lvl2_acc) / 2.0

    # Write results (append)
    with open(results_path, "a") as f:
        print(
            f"\n=== Result: use_dense={use_dense}, num_unfrozen_layers={num_unfrozen_layers} ===",
            file=f,
        )
        print(f"Epochs: {epochs}", file=f)
        print("Validation results:", file=f)
        print(val_results, file=f)
        print(
            f"Combined score (lvl1_acc + lvl2_acc) / 2 = {score:.6f}",
            file=f,
        )
        print("-" * 80, file=f)

    # Also print to stdout
    print("Validation results:", val_results)
    print(f"Combined score = {score:.6f}")


if __name__ == "__main__":
    main()
