import utils
import keras

callback1 = keras.callbacks.TensorBoard(
    log_dir="../logs/gate1", histogram_freq=0, write_graph=True, write_images=True
)
callback2 = keras.callbacks.TensorBoard(
    log_dir="../logs/gate2", histogram_freq=0, write_graph=True, write_images=True
)

strata_columns = ("model", "lighting")
(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_stratified_data(
    columns=strata_columns, target_size=(300, 300), strata_threshold=38
)

input_shape = train_x[0].shape


feature_extractor = keras.Sequential(
    [
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
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

x = feature_extractor(input)
gate1 = keras.Model(input, output_gate1(x))
gate2 = keras.Model(input, output_gate2(x))


gate1.compile(
    optimizer=keras.optimizers.Adam(6e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.Accuracy()],
)
gate2.compile(
    optimizer=keras.optimizers.Adam(6e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

gate1.fit(train_x, train_y['gate1'], epochs=5, batch_size=16)
other_filter = train_y["model"] != 'Other car'
gate2.fit(train_x[other_filter], train_y.loc[other_filter,
          'gate2'], epochs=5, batch_size=16)

print("Light")
filter = val_y["lighting"] == "Light"
gate1.evaluate(test_x=val_x[filter], test_y=val_y.loc[filter, "gate1"])
print("Medium")
filter = val_y["lighting"] == "Medium"
gate1.evaluate(test_x=val_x[filter], test_y=val_y.loc[filter, "gate1"])
print("Dark")
filter = val_y["lighting"] == "Dark"
gate1.evaluate(test_x=val_x[filter], test_y=val_y.loc[filter, "gate1"])
