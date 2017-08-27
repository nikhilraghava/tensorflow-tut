import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Model parameters
m = tf.Variable([0.3], dtype=tf.float32, name='m')
c = tf.Variable([-0.3], dtype=tf.float32, name='c')
# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = tf.add(tf.multiply(m, x), c)
y = tf.placeholder(tf.float32, name='y')

# Loss
# sum of the squares
loss = tf.reduce_sum(tf.square(tf.subtract(linear_model, y)))
# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01, name='train_min')
train = optimizer.minimize(loss)

# Training data
x_train = [1, 2, 3, 4]
y_train = [0, -2, -1, 10]
# Training loop
init = tf.global_variables_initializer()
# Reset values
sess = tf.Session()
sess.run(init)
# Train
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
# Evaluate training accuracy
curr_m, curr_c, curr_loss = sess.run([m, c, loss], {x: x_train, y: y_train})
print("m: %s c: %s loss: %s" % (curr_m, curr_c, curr_loss))
