# TensorFlow Tutorials - Softmax Regression

A vast majority of us got initiated into programming through the typical "Hello World." program where you just learn to print the phrase "Hello World." onto the terminal. Like programming, machine learning too has a "Hello World." program and it is called MNIST. The MNIST (Modified National Institute of Standards and Technology) dataset contains 70,000 images of hand written digits along with labels which tell us which image corresponds to which number. The numbers range from `0` to `9`. In this tutorial, we are going to train a model to look at images and predict what digits they are. The prediction is going to be a probability rather than a definitive prediction of class and to do that we will be using softmax regression. 

The MNIST dataset is hosted on [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/). The dataset contains 70,000 images as mentioned earlier and it's split into 55,000 data points of training data, 10,000 points of test data and 5,000 points of validation data. Every MNIST data point has two parts to it, an image of the handwitten digit and a corresponding label. In our model, the images are going to the "`x`" and the labels, "`y`". Both the training and the testing data set contains images and their corresponding labels. Each image is 28 pixel by 28 pixel and we can interpret the images as a big array of numbers. We can flatten this array into a vector of 784 numbers (28 x 28). You should note that flattening the array will result in the loss of infromation about the 2D struture of the images. Each number in the vector is a pixel intensity value between `0` and `1` (something like `0.4`, `0.9` and `0.7`). As mentioned earlier, each image in MNIST has a corresponding label between `0` and `9` and we would want our labels to be `one-hot` vectors. A one-hot vector is a vector which is `0` in most dimensions, and `1` in a single dimension. So if the label corresponding to an image is `3`, the one-hot vector is going to be `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` and if the label is `0` the one-hot vector is `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`. Basically,the value of the label corresponds to the index of `1` in the vector.

As mentioned earlier, we want our model to give us probablities instead of definitive predictions so we use softmax regression. Softmax gives us a list of probability values between `0` and `1` that adds up to `1`. Our softmax regression model has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities. To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. We also add some extra evidence called a bias to the weighted sum of pixel intensities. Biases basically make certain evidences independent of the input, so even if the weighted sum is 0, you still have some evidence that the image belongs/does not belong to a certain class. The result is that the evidence for a class *i* given an input *x* is:

<div align="center">
  <img src="https://cldup.com/ttvcgjROql.png"><br>
</div>

where `Wi` is the weights and `bi` is the bias for class `i`, and `j` is an index for summing over the pixels in our input image `x`. We then convert the evidence tallies into our predicted probabilities, `y` using the softmax function:

<div align="center">
  <img src="https://cldup.com/KDaO5ykUi7.png"><br>
</div>

Here softmax is serving as an activation function, shaping the output of our linear function into the form we want - in this case, a probability distribution over 10 classes and it is defined as:

<div align="center">
  <img src="https://cldup.com/y_VWl_UTcz.png"><br>
</div>

Expanding the equation, we get:

<div align="center">
  <img src="https://cldup.com/aMYqYFKHWD.png"><br>
</div>

Our softmax regression model can be pictured as looking something like the following, but with a lot more `x`s. 

<div align="center">
  <img src="https://www.tensorflow.org/images/softmax-regression-scalargraph.png"><br>
</div>