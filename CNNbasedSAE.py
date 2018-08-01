# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Visualize decoder setting
# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 28, 28, 1], name = "input")

# hidden layer settings
n_hidden_1 = 128 # 1st layer num features
n_hidden_2 = 30 # 2nd layer num features
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
}

# Building the encoder
def encoder(x):
    conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 28x28x16
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 14x14x16
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 14x14x8
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 7x7x8
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 7x7x8
    maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Encoder Hidden layer with sigmoid activation #1
    maxpool3 = tf.reshape(maxpool3, [-1, 128])
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(maxpool3, weights['encoder_h1']), biases['encoder_b1']))
    # # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_1, layer_2

# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'])), biases['decoder_b1']))
    # # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'])), biases['decoder_b2']))
    layer_2 = tf.reshape(layer_2, [-1, 4, 4, 8])
    upsample1 = tf.image.resize_images(layer_2, size=(7, 7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 7x7x8
    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 7x7x8
    upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 14x14x8
    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 14x14x8
    upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 28x28x8
    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 28x28x16
    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
    decoded = tf.nn.sigmoid(logits)
    # Encoder Hidden layer with sigmoid activation #1
    # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'])), biases['decoder_b1']))
    # # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'])), biases['decoder_b2']))
    return logits

def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho

def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

# Construct model
encoder_1, encoder_2= encoder(X)
decoder_1 = decoder(encoder_2)

# Prediction
y_pred = decoder_1
# Targets (Labels) are the input data.
y_true = X

results = [encoder_1,encoder_2]
Wset = [weights['encoder_h1'], weights['encoder_h2'],  tf.transpose(weights['encoder_h2']),  tf.transpose(weights['encoder_h1'])]
alpha = 5e-6
beta = 7.5e-6
rho = 0.06

"""
KL Divergence + L2 regularization
"""
kldiv_loss = reduce(lambda x,y : x+y, map(lambda x : tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
l2_loss = reduce(lambda x, y : x + y, map(lambda x : tf.nn.l2_loss(x), Wset))

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + alpha * l2_loss + beta * kldiv_loss
# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
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
            batch_xs = batch_xs[0].reshape([-1,28,28,1])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),

                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show].reshape([-1,28,28,1])})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='Greys_r')
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='Greys_r')
    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()
