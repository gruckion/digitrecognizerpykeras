import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from multi_layer_perceptron import MultiLayerPerceptron


def main():
    classifier = MultiLayerPerceptron()
    classifier.train_model()
    classifier.evaluate()


if __name__ == "__main__":
    main()
