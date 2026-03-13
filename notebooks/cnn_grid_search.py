import tensorflow as tf
import pandas as pd
import numpy as np
import utils
from keras import layers
import keras
from sklearn.metrics import f1_score
import itertools
import gc
import sys


SEED = 42
IMG_SIZE = (300, 300)
EPOCHS = 100
BATCH_SIZE = 32

AUGMENTATION_CONFIGS = [
    ("Flip", lambda: layers.RandomFlip("horizontal",    seed=SEED)),
    ("Rotation", lambda: layers.RandomRotation(0.05,    seed=SEED)),
    ("Zoom", lambda: layers.RandomZoom(0.1,             seed=SEED)),
    ("Contrast", lambda: layers.RandomContrast(0.2,     seed=SEED)),
    ("Brightness", lambda: layers.RandomBrightness(0.2, seed=SEED)),
]

LAYER_NAMES = [name for (name, _) in AUGMENTATION_CONFIGS]


(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_lvl1_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

with open('out.txt', 'w') as f:
    def get_combinations():
        all_combos = []
        for r in range(1, len(AUGMENTATION_CONFIGS) + 1):
            for combo in itertools.combinations(range(len(AUGMENTATION_CONFIGS)), r):
                all_combos.append(combo)
        return all_combos

    def build_dynamic_model(selected_indices):
        input_layer = keras.Input(shape=(*IMG_SIZE, 3))
        x = input_layer

        # Create fresh augmentation layers each time (no global layer instances)
        for i in selected_indices:
            _, layer_factory = AUGMENTATION_CONFIGS[i]
            x = layer_factory()(x)

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

        return keras.Model(inputs=input_layer, outputs={"lvl1": lvl1, "lvl2": lvl2})

    results = []
    best_f1 = -1
    best_config = None

    combinations = get_combinations()
    print(
        f"Starter testing av {len(combinations)} ulike augmentasjons-konfigurasjoner...")

    for combo_indices in combinations:
        current_names = [LAYER_NAMES[i] for i in combo_indices]
        config_str = " + ".join(current_names)
        print(f"\n[Tester]: {config_str}", file=f)

        model = build_dynamic_model(combo_indices)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=6e-4),
            loss={"lvl1": "binary_crossentropy",
                  "lvl2": "sparse_categorical_crossentropy"},
            metrics={"lvl1": [keras.metrics.BinaryAccuracy(name="acc")]},
            weighted_metrics={
                "lvl2": [keras.metrics.SparseCategoricalAccuracy(name="acc")]},
        )

        y = {'lvl1': train_y["lvl1"], 'lvl2': train_y["lvl2"]}

        n_samples = len(train_y["lvl1"])  # IMPORTANT: use number of samples
        sample_weight = {
            "lvl1": np.ones(n_samples, dtype=np.float32),
            "lvl2": train_y['lvl1'].to_numpy(dtype=np.float32),
        }

        validation_data = (
            val_x, {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]})

        model.fit(
            x=train_x,
            y=y,
            sample_weight=sample_weight,
            validation_data=validation_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        preds = model.predict(val_x, verbose=0)
        p1_probs = preds['lvl1'].flatten()
        y_pred = (p1_probs >= 0.5).astype(int)

        current_f1 = f1_score(val_y['lvl1'], y_pred)

        results.append({
            "Config": config_str,
            "F1_Score": current_f1
        })

        print(f"Resultat F1: {current_f1:.4f}", file=f, flush=True)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_config = config_str

        del config_str
        del model
        del current_names
        del y
        del validation_data
        del n_samples
        del sample_weight
        del preds
        del p1_probs
        del y_pred
        del current_f1
        keras.backend.clear_session()
        gc.collect()

    df_results = pd.DataFrame(results).sort_values(
        by="F1_Score", ascending=False)
    print("\n--- ENDELIG RANKING ---", file=f)
    print(df_results, file=f)

    print(
        f"\nBeste konfigurasjon: {best_config} med F1-score: {best_f1:.4f}", file=f)
