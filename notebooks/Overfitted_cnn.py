from keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import utils
import keras
import numpy as np
from aim.keras import AimCallback
import pandas as pd
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns


(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_lvl1_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2)
])


input = x = keras.Input(shape=(train_x[0].shape))
x = input
x = data_augmentation(x)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)


lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
lvl2 = layers.Dense(8, activation="softmax", name="lvl2")(x)

model = keras.Model(inputs=input, outputs={
                    "lvl1": lvl1, "lvl2": lvl2}, name="Overfitted_model")


opt = keras.optimizers.Adam(learning_rate=1e-5)

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


BATCH_SIZE = 32
EPOCHS = 100
SEED = 42
aim_cb = AimCallback(
    repo=".",
    experiment="Vehicle_lvl_Test",
)

hparams = {
    "learning_rate": opt.learning_rate,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "Adam",
    "model": "Overfitted_model",
}

sample_weight = {
    "lvl1": np.ones(len(train_y)),
    "lvl2": train_y['lvl1'].to_numpy()
}

validation_data = (val_x, {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]})

y = {'lvl1': train_y["lvl1"], 'lvl2': train_y["lvl2"]}
history = model.fit(epochs=EPOCHS, batch_size=BATCH_SIZE, x=train_x, y=y,
                    sample_weight=sample_weight, validation_data=validation_data, callbacks=[aim_cb, early_stop])


y = {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]}
model.evaluate(x=val_x, y=y)


def predict_heads(model, data_x):
    """Returns (lvl1_probs, lvl2_probs) for the overfitted model."""
    preds = model.predict(data_x, verbose=0)
    return preds['lvl1'].flatten(), preds['lvl2']


p1, p2 = predict_heads(model, val_x)
y1_true = val_y['lvl1'].to_numpy()
y2_true = val_y['lvl2'].to_numpy()


def eval_lvl1_by_lighting(y_true, p1_probs, lighting_series, threshold=0.5):
    rows = []
    y_pred = (p1_probs >= threshold).astype(int)

    for cond in ["Light", "Medium", "Dark"]:
        mask = (lighting_series == cond)
        if not mask.any():
            continue

        yt, yp = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        rows.append({
            "lighting": cond,
            "n_total": len(yt),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "TNR": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "lvl1_acc": accuracy_score(yt, yp),
            "lvl1_bal_acc": balanced_accuracy_score(yt, yp),
            "lvl1_f1": f1_score(yt, yp)
        })
    return pd.DataFrame(rows)


print("\n--- TABLE 1: LEVEL 1 BY LIGHTING ---")
print(eval_lvl1_by_lighting(y1_true, p1, val_y['lighting']))


def plot_lvl1_confusion_matrix(y_true, p1_probs, threshold=0.5):
    y_pred = (p1_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Other car', 'Tesla'],
                yticklabels=['Other car', 'Tesla'])
    plt.xlabel('Predikert')
    plt.ylabel('Faktisk (Fasit)')
    plt.title('Confusion Matrix: Level 1 (Vehicle Detection)')
    plt.savefig("confusion_matrix")
    plt.close()


plot_lvl1_confusion_matrix(y1_true, p1)
