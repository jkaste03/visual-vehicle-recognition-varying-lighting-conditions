import utils
import keras
import numpy as np

columns = ("model", "lighting", "year")
other = 'Other car'
(train_x, train_y), (test_x, test_y) = utils.read_stratified_data(
    columns=columns, strata_threshold=38)

vt = np.vectorize(lambda x: 1 if x else 0)

train_y_gate1 = train_y["model"] == 'Other car'
train_y_gate1 = vt(train_y_gate1)
test_y_gate1 = test_y["model"] == 'Other car'
test_y_gate1 = vt(test_y_gate1)

input_shape = train_x[0].shape
num_classes = 1


def model_to_int(model):
    dict = {'S': 0, '3': 1, 'X': 2, 'Y': 3}
    return dict[model]


train_y_gate2 = train_y["model"][train_y["model"] != other]
train_y_gate2 = np.vectorize(model_to_int)(train_y_gate2)

train_x_gate2 = train_x[train_y["model"] != other]

test_y_gate2 = test_y["model"][test_y["model"] != other]
test_y_gate2 = np.vectorize(model_to_int)(test_y_gate2)

test_x_gate2 = test_x[test_y["model"] != other]

input_shape = train_x[0].shape


def get_feature_extractor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    return keras.Model(inputs, x, name="extractor")


feature_extractor = get_feature_extractor(input_shape=input_shape)

input_gate1 = keras.Input(shape=input_shape)
x = feature_extractor(input_gate1)
outputs_gate1 = keras.layers.Dense(1, activation="sigmoid")(x)
model_a = keras.Model(input_gate1, outputs_gate1)

input_gate2 = keras.Input(shape=input_shape)
x = feature_extractor(input_gate2)
output_gate2 = keras.layers.Dense(4, activation="softmax")(x)
model_b = keras.Model(input_gate2, output_gate2)

model_a.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model_b.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_a.fit(train_x, train_y_gate1, epochs=10, batch_size=16)
print(model_a.evaluate(test_x, test_y_gate1))
model_b.fit(train_x_gate2, train_y_gate2, epochs=10, batch_size=16)


print(model_a.evaluate(test_x, test_y_gate1))
print(model_b.evaluate(test_x_gate2, test_y_gate2))
print(model_a.predict(test_x))
print(test_x[0].shape)
