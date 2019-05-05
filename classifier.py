from abc import abstractmethod
from numpy import ndarray
from digits_model import DigitsModel


class Classifier:
    """An interface a train and predict classifier"""

    @abstractmethod
    def train(self, model: DigitsModel) -> None:
        """Train the classifier on the set of observations"""
        pass
