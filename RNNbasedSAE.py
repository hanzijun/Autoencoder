# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.layers.core import Dense

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Visualize decoder setting
# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None,  n_input])

# hidden layer settings
n_hidden_0 = 384
n_hidden_1 = 200 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
weights = {
    'encoder_h0': tf.Variable(tf.random_normal([784, n_hidden_0])),
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_0])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b0': tf.Variable(tf.random_normal([n_hidden_0])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    x = tf.reshape(x, [-1, 1, 784])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(784, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_new = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(1)], state_is_tuple=True)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=None, dtype=tf.float32, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    layer_0 = outputs[-1]

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, weights['encoder_h0']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h1']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h2']), biases['encoder_b3']))
    return  layer_1, layer_2, layer_3

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_0 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'])), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, tf.transpose(weights['encoder_h1'])), biases['decoder_b0']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h0'])), biases['decoder_b2']))
    layer_22 = tf.reshape(layer_2, [-1,  1, 784])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(784, forget_bias=1.0, state_is_tuple=True, reuse = True)
    # lstm_cell_new = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(1)], state_is_tuple=True)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, layer_22, initial_state=None, dtype=tf.float32, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    layer_3 = outputs[-1]
    return layer_1, layer_2, layer_3

def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho

def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

# Construct model
encoder_1, encoder_2 , encoder_3= encoder(X)
decoder_1, decoder_2, decoder_3 = decoder(encoder_3)

# Prediction

y_pred = decoder_3
# Targets (Labels) are the input data.
y_true = X

results = [encoder_1, encoder_2, encoder_3]
Wset = [weights['encoder_h0'], weights['encoder_h1'], weights['encoder_h2'],  tf.transpose(weights['encoder_h2']),  tf.transpose(weights['encoder_h1']), tf.transpose(weights['encoder_h0'])]
alpha = 5e-6
beta = 7.5e-5
rho = 0.05

"""
KL Divergence + L2 regularization
"""
# kldiv_loss = reduce(lambda x,y : x+y, map(lambda x : tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
# l2_loss = reduce(lambda x, y : x + y, map(lambda x : tf.nn.l2_loss(x), Wset))

# Define loss and optimizer, minimize the squared error
# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + alpha * l2_loss + beta * kldiv_loss
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),

                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()
