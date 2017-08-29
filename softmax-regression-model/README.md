# TensorFlow Tutorials - Softmax Regression

A vast majority of us got initiated into programming through the typical "Hello World." program where you just learn to print the phrase "Hello World." onto the terminal. Like programming, machine learning too has a "Hello World." program and it is called MNIST. The MNIST (Modified National Institute of Standards and Technology) dataset contains 70,000 images of hand written digits along with labels which tell us which image corresponds to which number. The numbers range from `0` to `9`. In this tutorial, we are going to train a model to look at images and predict what digits they are. The prediction is going to be a probability rather than a definitive prediction of class and to do that we will be using softmax regression. 

The MNIST dataset is hosted on [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/). The dataset contains 70,000 images as mentioned earlier and it's split into 55,000 data points of training data, 10,000 points of test data and 5,000 points of validation data. Every MNIST data point has two parts to it, an image of the handwitten digit and a corresponding label. In our model, the images are going to the "`x`" and the labels, "`y`". Both the training and the testing data set contains images and their corresponding labels. Each image is 28 pixel by 28 pixel and we can interpret the images as a big array of numbers. We can flatten this array into a vector of 784 numbers (28 x 28). You should note that flattening the array will result in the loss of infromation about the 2D struture of the images. Each number in the vector is a pixel intensity value between `0` and `1` (something like `0.4`, `0.9` and `0.7`). As mentioned earlier, each image in MNIST has a corresponding label between `0` and `9` and we would want our labels to be `one-hot` vectors. A one-hot vector is a vector which is `0` in most dimensions, and `1` in a single dimension. So if the label corresponding to an image is `3`, the one-hot vector is going to be `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` and if the label is `0` the one-hot vector is `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`. Basically,the value of the label corresponds to the index of `1` in the vector.

As mentioned earlier, we want our model to give us probablities instead of definitive predictions so we use softmax regression. Softmax gives us a list of probability values between `0` and `1` that adds up to `1`. Our softmax regression model has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities. To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. We also add some extra evidence called a bias to the weighted sum of pixel intensities. Biases basically make certain evidences independent of the input, so even if the weighted sum is 0, you still have some evidence that the image belongs/does not belong to a certain class. The result is that the evidence for a class *i* given an input *x* is:

<div align="center">
<br><img src="https://cldup.com/ttvcgjROql.png" width="290" height="56.5"><br><br>
</div>

where `Wi` is the weights and `bi` is the bias for class `i`, and `j` is an index for summing over the pixels in our input image `x`. We then convert the evidence tallies into our predicted probabilities, `y` using the softmax function:

<div align="center">
<br><img src="https://cldup.com/KDaO5ykUi7.png" width="236.5" height="25"><br><br>
</div>

Here softmax is serving as an activation function, shaping the output of our linear function into the form we want - in this case, a probability distribution over 10 classes and it is defined as:

<div align="center">
<br><img src="https://cldup.com/y_VWl_UTcz.png" width="339.5" height="25"><br><br>
</div>

Expanding the equation, we get:

<div align="center">
<br><img src="https://cldup.com/aMYqYFKHWD.png" width="278" height="63.5"><br><br>
</div>

Our softmax regression model can be pictured as looking something like the following, but with a lot more `x`s. 

<div align="center">
<br><img src="https://www.tensorflow.org/images/softmax-regression-scalargraph.png" width="581.4" height="232.2"><br><br>
</div>

In summary, our model can be written as:

<div align="center">
<br><img src="https://cldup.com/eWN_cXvr7r.png" width="272" height="30.5"><br><br>
</div>

Vizualizing the above equation in terms of vectors, we get:

<div align="center">
<br><img src="https://www.tensorflow.org/images/softmax-regression-vectorequation.png" width="524.8" height="128"><br><br>
</div>

Now that we have defined our entire model in mathematical terms, let's start coding.

## Implementing the Softmax Regression Model

First we need to import TensorFlow.

```python
import tensorflow as tf
```

Now we need to obtain the MNIST dataset. Thankfully, TensorFlow has an inbulit function which allows us to get the MNIST dataset, extract it and use it, they have even split the dataset for us so we don't have to do it ourselves.

```python
from tensorflow.examples.tutorials.mnist import input_data

# Download and extract the MNIST data set, convert to one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
``` 

Here we download our data and convert the labels to a one-hot vector. Now let's reserve a place for our images.

```python
# MNIST images, each flattened into a 784-dimensional vector
x = tf.placeholder(tf.float32, [None, 784])
```

`x` is a `placeholder`, a value that we'll input when we ask TensorFlow to run a computation. We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector. We represent this as a 2-D `Tensor` of floating-point numbers, with a shape `[None, 784]`. `x` is of shape `[None, 784]` and we will be feeding in images in batches so `None` here is able to support any batch size that we specify. Now we need our weights and biases. Like in the [linear regression model example](https://github.com/nikhilraghava/tensorflow-tut/tree/master/linear-regression-model), weights and biases will be like our `m` and `c`, they will be variables whose values will be constantly updated as we train the model. Let's declare weights and biases in our code as variables.

```python
# Weights
W = tf.Variable(tf.zeros([784, 10]))
# Biases
b = tf.Variable(tf.zeros([10]))
```

Our weights, `w` and biases, `b` will be `Tensors` full of `0`s with a shape of `[784, 10]` and `[10]` respectively. Notice that `W` has a shape of `[784, 10]` because we want to matrix multiply the 784-dimensional image vectors with it to produce 10-dimensional vectors of evidence for the different classes of labels. `b` has a shape of `[10]` so we can add it to the result of the matrix multiplication. Now let us define our model.

```python
# Model
y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))
```

The code above is just following the equation of our model that we defined earlier. I prefer to use `tf.add` to add two `Tensors` together, adding them using the regular `+` would also work. Like our linear regression model we also need to define a loss/cost function and for this model we will be using a very common loss function called cross-entropy. Cross-entropy is defined as:

<div align="center">
<br><img src="https://cldup.com/2A5LB8bwxV.png" width="242" height="51"><br><br>
</div>