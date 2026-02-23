import utils
import keras
from Hierarchical_model import Hierarchical_model


callback1 = keras.callbacks.TensorBoard(
    log_dir="./logs/gate1", histogram_freq=0, write_graph=True, write_images=True
)
callback2 = keras.callbacks.TensorBoard(
    log_dir="./logs/gate2", histogram_freq=0, write_graph=True, write_images=True
)

strata_columns = ("model", "lighting")
(train_x, train_y), (test_x, test_y) = utils.read_stratified_data(
    columns=strata_columns)

input_shape = train_x[0].shape


feature_extractor = keras.Sequential(
    [
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4)
    ]
)

input = keras.Input(shape=input_shape)
output_gate1 = keras.Sequential([
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

output_gate2 = keras.Sequential([
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(7, activation="softmax"),
])

model = Hierarchical_model(input=input,
                           feature_extractor=feature_extractor,
                           output_gate1=output_gate1,
                           output_gate2=output_gate2,
                           callbacks=([callback1], [callback2]))

model.compile_gate1(
    optimizer=keras.optimizers.Adam(6e-4),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
model.compile_gate2(
    optimizer=keras.optimizers.Adam(6e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_x, train_y, epochs=5, batch_size=16)
model.evaluate(test_x=test_x, test_y=test_y)
