import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


class MultiLayerPerceptron():
    """simple multi-layer perceptron model"""

    def __init__(self):
        # fix random seed for reproducibility
        numpy.random.seed(7)

        # load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # rehspaae the 28*28 images to a flattened 784 vector for each image
        self.num_pixels: int = self.X_train.shape[1] * self.X_train.shape[2]
        self.X_train: float = self.X_train.reshape(
            self.X_train.shape[0], self.num_pixels).astype('float32')
        self.X_test: float = self.X_test.reshape(
            self.X_test.shape[0], self.num_pixels).astype('float32')

        # normalize grey scale pixels from 0-255 to 0-1
        self.X_train: float = self.X_train / 255
        self.X_test: float = self.X_test / 255

        # one hot encode outputs of the multi-class 0-9 set
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]

    def train_model(self) -> None:
        # build the model
        model = self._baseline_model()

        # Fit the model
        self.model.fit(self.X_train, self.y_train, validation_data=(
            self.X_test, self.y_test), epochs=10, batch_size=200, verbose=2)

    def evaluate(self) -> None:
        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    def _baseline_model(self) -> None:
        # create model
        self.model = Sequential()
        self.model.add(Dense(
            self.num_pixels,
            input_dim=self.num_pixels,
            kernel_initializer='normal',
            activation='relu'))

        self.model.add(Dense(
            self.num_classes,
            kernel_initializer='normal',
            activation='softmax'))

        # Compile model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])
