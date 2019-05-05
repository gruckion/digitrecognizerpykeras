from numpy import ndarray
from keras.models import Sequential
from typing import Tuple


class Evaluator():
    @staticmethod
    def evaluate(classifier: Sequential, data: Tuple[ndarray, ndarray]) -> None:
        (x_train, y_train), (x_test, y_test) = data
        # Final evaluation of the model
        scores = classifier.evaluate(x_test, y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))
