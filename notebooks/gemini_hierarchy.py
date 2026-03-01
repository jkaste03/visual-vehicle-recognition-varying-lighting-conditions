import keras
from keras import layers

import utils
strata_columns = ("model", "lighting")
(train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.read_stratified_data(
    columns=strata_columns, target_size=(300, 300), strata_threshold=38
)

# 1. Input Layer
input_layer = layers.Input(shape=(300, 300, 3))

# 2. Shared Feature Extractor (The "Knowledge")
x = layers.Conv2D(128, (3, 3), activation="relu")(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
shared_features = layers.Flatten()(x)

# 3. Gate 1: The Binary Decision (Is it a car?)
# Activation must be sigmoid to stay between 0 and 1
gate1_out = layers.Dense(1, activation="sigmoid",
                         name="gate1")(shared_features)

# 4. The "Power Switch" (Hierarchical Link)
# This multiplies the shared features by the probability from Gate 1
gated_features = layers.Multiply()([shared_features, gate1_out])

# 5. Gate 2: The Specialist (Which car model?)
# It only receives strong signals if gate1_out is close to 1.0
x2 = layers.Dense(10, activation="relu")(gated_features)
gate2_out = layers.Dense(7, activation="softmax", name="gate2")(x2)

# Define the Model
hierarchical_model = keras.Model(
    inputs=input_layer, outputs=[gate1_out, gate2_out])

# Compile
hierarchical_model.compile(
    optimizer=keras.optimizers.Adam(6e-4),
    loss={
        "gate1": "binary_crossentropy",
        "gate2": "sparse_categorical_crossentropy"
    },
    metrics={"gate1": "accuracy", "gate2": "accuracy"}
)

y_train_dict = {
    "gate1": train_y['gate1'].values,  # Binary: 0 or 1
    "gate2": train_y['gate2'].values  # Integer: 0 through 6
}


hierarchical_model.fit(train_x, y_train_dict)
