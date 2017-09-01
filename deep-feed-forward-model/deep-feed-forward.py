import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Download and extract the MNIST data set, convert to one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hidden layers
n_nodes_hl1 = 800
n_nodes_hl2 = 800
n_nodes_hl3 = 800

# Classes
n_classes = 10
# Batch size
batch_size = 100

# Variables
# Matrix: height x width
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)


def neural_network_model(data):
    # (input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Model
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    # Feed input to model
    prediction = neural_network_model(x)
    # Cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # Optimizer (Adam)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Model will train for 10 cycles, feed forward + backprop
    epochs = 10

    # Session
    with tf.Session() as sess:
        # Initialize global variable
        sess.run(tf.global_variables_initializer())
        # Run cycles
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss', epoch_loss)

        # Check prediction
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("{0:f}%".format(result * 100))


train_neural_network(x)