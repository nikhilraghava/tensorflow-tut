# TensorFlow Tutorials - Linear Regression

In this tutorial, we will be looking at how we can use TensorFlow to implement a linear regression model on a given data set. In our case, the data set is going to be very, very small compared to real life data sets that we will be looking at later in the series (our data set only has 4 points). Most of us just get hit by the term linear regression (if you have not taken some advanced math classes), linear regression, in layman's terms is all about drawing the best fit line that best describes a given data set. So now that we know what linear regression is, let's get started.

First, make sure you have TensorFlow installed on your computer and import it as follows. 

```python
import tensorflow as tf
```

Our ultimate goal as stated earlier is to derive the best fit line that best describes the given data set. A line, as we all know follows the standard equation, `y = mx + c`. In this equation, we have two variables that are of concern to us, `m` and `c`. In our training data `y` and `x` would be given to us so we don't have to care about them for now. So let's go ahead and declare `m` and `c` as a variable.

```python
m = tf.Variable([0.3], dtype=tf.float32)
c = tf.Variable([-0.3], dtype=tf.float32)
```

Here we the initialize the variables, `m` and `c` using the `tf.Variable()` constructor. We pass two arguments into the constructor, the first argument is the initial value of the variable, a `Tensor` and the second argument is the data type of the variable. Now we need to declare `y` and `x`, although `y` and `x` values are given to us in our data set, we need to reserve a place for them in our code. 

```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
```

`y` and `x` are what we call `placeholders`. `placeholders` as the name suggests, allows us to insert a placeholder for a `Tensor` that will be fed with a dictionary which contains the training data. Now, let us declare the linear model which is essentially the standard equation of the best fit line.

```python
linear_model = tf.add(tf.multiply(m, x), c)
```

Here we `tf.multiply`, `m` and `x` and `tf.add` them to `c`. This is exactly the same as `y = mx + c`. Now we need to write a loss function. A loss function or a cost function tells us how far apart the current model is from the provided data. We will be using a standard loss model for linear regression, which essentially sums the squares of the deltas between the current model and the provided data. 

```python
loss = tf.reduce_sum(tf.square(tf.subtract(linear_model, y)))
```

`linear_model - y` creates a vector where each element is the corresponding example's error delta. We use `tf.square` to square the deltas and use `tf.reduce_sum` to create a single scalar that abstracts the error of all examples. We originally declared our `m` and `c` variables to have an initial value of `0.3` and `-0.3` respectively and this value is not going to produce the best fit line for every data set that is going to be thrown at our linear regression model, so we need to adjust the `m` and `c` values to get a best fit line that will adapt and describe any data set that is going to be thrown at us. To do that we use an optimizer. The optimizer we are going to use is called a gradient descent optimizer. It essentially modifies each variable according to the magnitude of the derivative of loss with respect to that variable. There are other optimizers such as the `AdamOptimizer` that you can use. Different optimizers will give you different loss values at the end of the training session and the process of selecting the most suitable optimizer for your model is more of a trial and error process. Now let's implement the optimizer.

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

We pass the learning rate, `0.01` into the `tf.train.GradientDescentOptimizer` class. Then we minimize our `loss` using `optimizer.minimize`. Now we need to provide our training data.

```python
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
```

The training data that we have here is really, really small and in the real world this is not going to be the case. Now we need to initialize our global variables the TensorFlow way. 

```python
init = tf.global_variables_initializer()
```

Now lets run the session and train it.

```python
sess = tf.Session()
sess.run(init)
# Train for 1000 epochs
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
```

`sess.run` runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every operation and evaluate every `Tensor` in `fetches`, substituting the values in `feed_dict` for the corresponding input values. `fetches` are the first argument of `sess.run` and the dictionary corresponding the `feed_dict` argument is our training data, `{x: x_train, y: y_train}`. Finally, to evaluate the training accuracy,

```python
curr_m, curr_c, curr_loss = sess.run([m, c, loss], {x: x_train, y: y_train})
print("m: %s c: %s loss: %s" % (curr_m, curr_c, curr_loss))
```

Since `m` and `c` are `placeholder` variables, they would have been optimized to give us a best fit line that best describes our data at the end of the training session, so we can get the final values of `m`, `c` and the `loss` using `sess.run` since they are `Tensors`. And that's it, this is our linear regression model. Running the model should give you something close to the following result.

```terminal
m: [-0.9999969] c: [ 0.99999082] loss: 5.69997e-11
```

Using TensorBoard we get the following graph of our model.

![graph](https://cldup.com/XT5pcT4YUk.png)


## The Mathematics of Gradient Descent

In this section, I will be discussing with you the mathematics behind the gradient descent optimizer so that you can have a clear understanding of how our linear regression model works. The equation of our best fit line as you all know is `y = mx + c` and the `y` value is going to change for different values of `x`, so let's write that down as an equation.

<div align="center">
  <img src="https://cldup.com/awVh-0Y8bF.png"><br><br>
</div>

Now we need to find the error margin between the actual `y` value and the predicted `y` value. To do that we use the following equation. Where `y` is the actual value and `y hat` is the predicted value.

<div align="center">
  <img src="https://cldup.com/4y4Kkk29bs.png"><br><br>
</div>

The loss, as stated earlier, is calculated by summing the squares of the error deltas together.

<div align="center">
  <img src="https://cldup.com/nW8j3G5vAV.png"><br><br>
</div>

Substituting the standard equation of a best fit line into the loss function, we obtain the following equation.

<div align="center">
  <img src="https://cldup.com/gU651jHogq.png"><br><br>
</div>

Further substitution gives us the following equation of the loss function.

<div align="center">
  <img src="https://cldup.com/trJQp7Nr9n.png"><br><br>
</div>

The ultimate goal of gradient descent in our linear regression model is to minimize the loss value obtained from our loss function for different values of `m` and `c` during training. Before we derive our next equation, let us make `m` and `c` equal to `M` independently, so `c = M` and `m = M` this will make our derivation much simpler. From the equation above we can tell that any changes made to `m` or `c` will have a direct impact on the loss, since we generalized and equated `c` and `m` to `M` independently, any changes to `M` will impact the loss value, letting the loss value equal `L`. Since our loss function is essentially the sum of squared error deltas, the graph of `L` against `M` would give us an inverse parabolic curve. The loss is `0` or near `0` at the bottom of the curve so we need to make our way there by adjusting the values of `m` and `c` but using brute force will take a very, very long time, so we use gradient descent.

First, we need to have a derivative of the loss function so that we can see if we need to increase the value of `M` or decrease the value of `M`. The goal is to make the derivative of the loss function equal a value close to `0`. Since we are changing the value of `M` individually as we work, the derivative is a partial derivative.

<div align="center">
  <img src="https://cldup.com/4zK_JUa75E.png"><br><br>
</div>

From the derivative of our loss function, we can tell if we need to increase or decrease the value of `M`. If the derivative is positive, going uphill, increasing the value of `M` would increase the loss, so we need to decrease it. If the derivative is negative, going downhill, increasing the value of `M` would decrease the loss. Now that we know which way to go we need to make a move. To update the value of `M` we use the following formula.

<div align="center">
  <img src="https://cldup.com/kVeMFrBnl1.png"><br><br>
</div>

In the above formula, `alpha` is the learning rate, the argument of the `tf.train.GradientDescentOptimizer` class. This process of updating and calculating the loss happens till the derivative of the loss function is as close to `0` as possible. In our model, `m` and `c` are updated independently and simultaneously. This is how gradient descent works.