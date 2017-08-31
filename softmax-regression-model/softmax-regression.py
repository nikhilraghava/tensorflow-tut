import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Download and extract the MNIST data set, convert to one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# MNIST images, each flattened into a 784-dimensional vector
x = tf.placeholder(tf.float32, [None, 784])
# Weights
W = tf.Variable(tf.zeros([784, 10]))
# Biases
b = tf.Variable(tf.zeros([10]))

# Model
y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

# Actual label
y_ = tf.placeholder(tf.float32, [None, 10])
# Cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# Gradient descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch model in interactive session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Train model for 1000 epochs
for _ in range(1000):
    # Batch of 100 random training data points
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Check if prediction matched the value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Cast booleans to floats
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Check accuracy on test data
result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("{0:f}%".format(result * 100))