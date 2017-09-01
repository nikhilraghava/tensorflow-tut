import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn 

# Download and extract the MNIST data set, convert to one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Epochs
epochs = 3
# Classes
n_classes = 10
# Batch size
batch_size = 128
# Chunck
chunck_size = 28
n_chunks = 28
# RNN size 
rnn_size = 128

# Variables
x = tf.placeholder(tf.float32, [None, n_chunks, chunck_size])
y = tf.placeholder(tf.float32)


def recurrent_neural_network(x):
    # Layer
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    # Transpose, reshape and split
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunck_size])
    x = tf.split(x, n_chunks, 0)
    # LSTM
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Output
    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def train_neural_network(x):
    # Feed input to model
    prediction = recurrent_neural_network(x)
    # Cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # Optimizer (Adam)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Session
    with tf.Session() as sess:
        # Initialize global variable
        sess.run(tf.global_variables_initializer())
        # Run cycles
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Reshape
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunck_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss', epoch_loss)

        # Check prediction
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result = sess.run(accuracy, feed_dict={x: mnist.test.images.reshape((-1, n_chunks, chunck_size)), y: mnist.test.labels})
        print("{0:f}%".format(result * 100))


train_neural_network(x)