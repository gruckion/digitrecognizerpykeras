### Digit Recognition Multi Layer Perceptron

This repo follows the guide on [multi class classifier for Digit Recognition](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/) using a multi layer perceptron in Python. This is on the MNist dataset loaded through Keras.

# Keywords and linked articles for learning
- Perceptron model

`A simple perceptron consists of an input and output layer with weights and a bias. This is capable of binary classification since it is only capable of classifying items between a linear decision boundary.`

- Multi-layer perceptron

`In addition to input and output layers the multi-layer perceptron has one or more hidden layers.`

- Single hidden layer

`The hidden layer allows for non-linear classification. Since a digit recognition system has 10 classes a hidden layer is needed to allow for non-linear classifications.`

- [np.reshape()](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.reshape.html)

`For reshaping multi dimensional arrays`

- multi-class classification

`A perceptron is a binary classifer, multi-class classification requires non-linear decision boundaries.
- One hot encoding`

`One hot encoding is the process of transforming your mutually exclusive catagoracal data into a feture vectors of the same size {1, 0, 0} {0, 1, 0}. If we use integer encoding e.g. `x = 1` or `x = 2` this will lead to results where we get a predicion mid way between the two values. Clearly with digits a two is always a two and is not a three or a nine. We do have percentage certainities on each but a cataegorical value can not belong to two classes, there is no ordinal relationship. This exclusive relationship can be easily described with one hot encoding.`

- Limitations of one hot encodidng

`The main limiation is that the representation size will grow with the corpus. With digits the corpus contains 10 classes, but with natura language this this could easily become 50 million. In this case we would then want to use distributed encoding.
- Rectifier activation function for neurons in hidden layer
Each layer in a neural network will have an activation function, this is a treshhold that must be achieved for the ouput to be classified in the range of [0,1]. For the rectifier activation function (c) below, this is the  se ction where the value is non zero. (a) is the sigmoid and (b) is the tanh.`

<img src="https://www.researchgate.net/profile/Wing_Ng8/publication/260525214/figure/fig1/AS:614085599182885@1523420824926/a-The-sigmoid-b-the-tanh-c-the-rectifier-activation-functions.png" height="150" />

- Softmax (softplus) activation function 
`Used on the output layer to turn outputs into probability-like values for more complex networks.`

<img src="https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png" height="200" />

- Other activation functions

<img src="https://cdn-images-1.medium.com/max/1600/1*p_hyqAtyI8pbt2kEl6siOQ.png" height="350" />

- Logarithmic loss function (Keras categorical_crossentrpoy)

`A loss function is used for back propergation when updating the weights within the multiple layers of a perceptron. For a single perceptron the output is a function of the input and the weights are updated to get to the desired output. In a MLP the output is allowed to change along with the weights whilst holding the inputs constant, this allows the weights to adjust by an amount represented by the loss function. The loss function is the amount by which the output is off from the target vallue. The type of loss function depends on the problem and varies for catagorical / continuous classificationsl.`

- ADAM gradient descent algorithm for learning weights

`When updating the weights in an MLP calculus is used to get the gradient at a point to see which direction to move to head towards a minimum. This may or may not be a local minima. This process is called gradient decent. By minimising the loss function above through gradient decent the weights can be adjusted to get the correct target output.`

- Model fitting epochs- [False 100% accuracy using Kaggle](https://www.kaggle.com/cdeotte/mnist-perfect-100-using-knn)

`When adjusting the weights to improve the accuracy of a learning algorithm multiple parses will be required to minimise the loss function. The number of parses to get to the minima is known as epochs.`

<img src="https://cdn-images-1.medium.com/max/1800/1*pwPIG-GWHyaPVMVGG5OhAQ.gif" height="250" />

- Epoch vs Batch size vs itteration

`We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch. More [here](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)`

- [Stats on the best performing model on MNIST]

(http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
