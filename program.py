from multi_layer_perceptron import MultiLayerPerceptron
from data_loader import DataLoader
from evaluator import Evaluator
from digits_model import DigitsModel


def main():
    model: DigitsModel = DataLoader.fetch_data()

    mlp = MultiLayerPerceptron()
    mlp.train(model)

    Evaluator.evaluate(mlp.classifier, model.data)


if __name__ == "__main__":
    main()
