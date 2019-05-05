from keras.utils import np_utils
from keras.datasets import mnist
from typing import Tuple
from numpy import ndarray
from digits_model import DigitsModel


class DataLoader():

    @staticmethod
    def fetch_data() -> DigitsModel:
        """Load in the MNIST data set from keras"""
        return DataLoader._reshape_data(mnist.load_data())

    @staticmethod
    def _reshape_data(data: Tuple[ndarray, ndarray]) -> DigitsModel:
        """Reshape the data"""
        (x_train, y_train), (x_test, y_test) = data

        # rehspape the 28*28 images to a flattened 784 vector for each image
        num_pixels: int = x_train.shape[1] * x_train.shape[2]
        x_train = x_train.reshape(
            x_train.shape[0], num_pixels).astype('float32')
        x_test = x_test.reshape(
            x_test.shape[0], num_pixels).astype('float32')

        # normalize grey scale pixels from 0-255 to 0-1
        x_train = x_train / 255
        x_test = x_test / 255

        # one hot encode outputs of the multi-class 0-9 set
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]

        data = (x_train, y_train), (x_test, y_test)
        return DigitsModel(data, num_pixels, num_classes)
