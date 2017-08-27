import tensorflow as tf

# Model parameters
m = tf.Variable([0.3], dtype=tf.float32, name='m')
c = tf.Variable([-0.3], dtype=tf.float32, name='c')
# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = tf.add(tf.multiply(m, x), c)
y = tf.placeholder(tf.float32, name='y')

# Loss
loss = tf.reduce_sum(tf.square(tf.subtract(linear_model, y)))  # sum of the squares
# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01, name='train_min')
train = optimizer.minimize(loss)

# Training data
x_train = [1, 2, 3, 4]
y_train = [1, 2, 3, 4]
# Training loop
init = tf.global_variables_initializer()
# Run the session and produce Tensorboard graph
# tensorboard --logdir="./graphs"
with tf.Session() as sess:
    # Tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # Reset values
    sess.run(init)
    # Train
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})
    # Evaluate training accuracy
    curr_m, curr_c, curr_loss = sess.run([m, c, loss], {x: x_train, y: y_train})
    print("m: %s c: %s loss: %s" % (curr_m, curr_c, curr_loss))
# Close writer
writer.close()
