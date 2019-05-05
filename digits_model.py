from typing import Tuple
from numpy import ndarray


class DigitsModel():
    def __init__(self, data: Tuple[ndarray, ndarray], num_pixels: int, num_classes: int):
        self.data = data
        self.num_pixels = num_pixels
        self.num_classes = num_classes

    def hello(self):
        return ""
