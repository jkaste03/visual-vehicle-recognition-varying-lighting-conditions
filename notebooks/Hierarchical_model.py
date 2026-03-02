import utils
import numpy as np
import keras
from typing import Any


class Hierarchical_model:
    gate1: keras.Model
    gate2: keras.Model
    other = 'Other car'

    def __init__(self, feature_extractor: keras.Model, input: keras.Input, output_gate1, output_gate2, callbacks=None):
        x = feature_extractor(input)
        self.gate1 = keras.Model(input, output_gate1(x))
        self.gate2 = keras.Model(input, output_gate2(x))
        self.compile_gate1 = self.gate1.compile
        self.compile_gate2 = self.gate2.compile
        self.callbacks = callbacks

    def fit_gate1(self, train_x, train_y, epochs=10, batch_size=16, validation_data=None):
        self.gate1.fit(train_x, train_y,
                       epochs=epochs, batch_size=batch_size, callbacks=self.callbacks[0], validation_data=validation_data)

    def fit_gate2(self, train_x, train_y, epochs=10, batch_size=16, validation_data=None):
        self.gate2.fit(train_x, train_y,
                       epochs=epochs, batch_size=batch_size, callbacks=self.callbacks[1], validation_data=validation_data)

    def fit(self, train_x, train_y, epochs=10, batch_size=16):
        self.fit_gate1(train_x=train_x, train_y=train_y["gate1"],
                       epochs=epochs, batch_size=batch_size)
        self.fit_gate2(train_x=train_x, train_y=train_y.loc[train_y["model"] != 'Other car', "gate2"],
                       epochs=epochs, batch_size=batch_size)

    def evaluate(self, test_x, test_y, ):

        print(self.gate1.evaluate(
            test_x, test_y["gate1"], callbacks=self.callbacks[0]))
        test_y_gate2 = test_y["gate2"][test_y["model"] != self.other]
        test_x_gate2 = test_x[test_y["model"] != self.other]

        print(self.gate2.evaluate(test_x_gate2,
              test_y_gate2, callbacks=self.callbacks[1]))

    def predict_gate1(self, x):
        return self.gate1.predict(np.expand_dims(x, axis=0))

    def predict_gate2(self, x):
        return self.gate2.predict(np.expand_dims(x, axis=0))

    def predict(self, x):
        pred1 = self.predict_gate1(x)
        if pred1 == 1:
            return pred1

        return self.predict_gate2(x)
