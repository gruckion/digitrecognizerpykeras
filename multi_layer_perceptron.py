import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from classifier import Classifier
from typing import Tuple
from numpy import ndarray
from digits_model import DigitsModel


class MultiLayerPerceptron(Classifier):
    """simple multi-layer perceptron model"""

    def __init__(self):
        # fix random seed for reproducibility
        numpy.random.seed(7)

    def train(self, model: DigitsModel) -> None:
        (x_train, y_train), (x_test, y_test) = model.data

        # build the model
        self._baseline_model(model.num_pixels, model.num_classes)

        # # Fit the model
        self.classifier.fit(x_train, y_train, validation_data=(
            x_test, y_test), epochs=10, batch_size=200, verbose=2)

    def _baseline_model(self, num_pixels: int, num_classes: int) -> None:
        # create model
        self.classifier = Sequential()
        self.classifier.add(Dense(
            num_pixels,
            input_dim=num_pixels,
            kernel_initializer='normal',
            activation='relu'))

        self.classifier.add(Dense(
            num_classes,
            kernel_initializer='normal',
            activation='softmax'))

        # Compile model
        self.classifier.compile(
            loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])
