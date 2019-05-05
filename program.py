import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# simple multi-layer perceptron model
def main():
    init_random(7)

    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # rehspaae the 28*28 images to a flattened 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train: float = X_train.reshape(
        X_train.shape[0], num_pixels).astype('float32')
    X_test: float = X_test.reshape(
        X_test.shape[0], num_pixels).astype('float32')

    # normalize grey scale pixels from 0-255 to 0-1
    X_train: float = X_train / 255
    X_test: float = X_test / 255

    # one hot encode outputs of the multi-class 0-9 set
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # build the model
    model = baseline_model(num_pixels, num_classes)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=10, batch_size=200, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))


def init_random(seed: int):
    # fix random seed for reproducibility
    numpy.random.seed(seed)

# define baseline model


def baseline_model(num_pixels: int, num_classes: int):
    # create model
    model = Sequential()
    model.add(Dense(
        num_pixels,
        input_dim=num_pixels,
        kernel_initializer='normal',
        activation='relu'))

    model.add(Dense(
        num_classes,
        kernel_initializer='normal',
        activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
