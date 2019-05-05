from multi_layer_perceptron import MultiLayerPerceptron


def main():
    classifier = MultiLayerPerceptron()
    classifier.train_model()
    classifier.evaluate()


if __name__ == "__main__":
    main()
