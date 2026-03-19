import utils
import keras
from keras import layers
import numpy as np
from sklearn.metrics import f1_score


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
EPOCHS = 150
BATCH_SIZE = 32
FILE_NAME = "Faiga_model"
MODEL_FOLDER = "./models/"


def build_model():
    input = x = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    lvl1 = layers.Dense(1, activation="sigmoid", name="lvl1")(x)
    lvl2 = layers.Dense(7, activation="softmax", name="lvl2")(x)

    return keras.Model(inputs=input, outputs={"lvl1": lvl1, "lvl2": lvl2}, name="v4_model")


opt = keras.optimizers.Adam(learning_rate=6e-4)


model = build_model()

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

model.summary()


(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_andre_data()

val_f1_cb = ValF1Callback(
    val_x=val_x,
    val_y_lvl1=val_y["lvl1"],
    val_y_lvl2=val_y["lvl2"],
    name_lvl1="val_f1_lvl1",
    name_lvl2="val_f1_lvl2_macro",
    name_combined="val_score_50_50",
    threshold=0.5,
)

train_y_dict = {'lvl1': train_y["lvl1"], 'lvl2': train_y["lvl2"]}
val_y_dict = {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]}
validation_data = (val_x, {'lvl1': val_y["lvl1"], 'lvl2': val_y["lvl2"]})

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_score_50_50',
    mode='max',
    patience=10,
    restore_best_weights=True
)

sample_weight = {
    "lvl1": np.ones(len(train_y)),
    "lvl2": train_y['lvl1'].to_numpy()
}

model.fit(x=train_x, y=train_y_dict, batch_size=BATCH_SIZE, callbacks=[val_f1_cb,
          early_stop], sample_weight=sample_weight, validation_data=validation_data, epochs=EPOCHS)
model.save(MODEL_FOLDER + FILE_NAME)
