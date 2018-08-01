# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return pool, argmax

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = 'SAME')
def max_unpool_2x2(x, shape):
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1]*2, shape[2]*2]))
    return inference

w_initializer = tf.random_normal_initializer(0., 0.1)
b_initializer = tf.constant_initializer(0.01)
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, 784])
x_origin = tf.reshape(x, [-1, 28, 28, 1])
W_e_conv1 = tf.get_variable('we1', [5,5,1,16], initializer=w_initializer)
b_e_conv1 = tf.get_variable('be1', [16, ], initializer=b_initializer)
h_e_conv1 = tf.nn.relu(tf.add(conv2d(x_origin, W_e_conv1), b_e_conv1))
h_e_pool1, argmax_e_pool1 = max_pool_2x2(h_e_conv1)

W_e_conv2 = tf.get_variable('we2', [5,5,16,32], initializer=w_initializer)
b_e_conv2 = tf.get_variable('be2', [32, ], initializer=b_initializer)
h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_pool1, W_e_conv2), b_e_conv2))
h_e_pool2, argmax_e_pool2 = max_pool_2x2(h_e_conv2)
code_layer = h_e_pool2
print("code layer shape : %s" % h_e_pool2.get_shape())


W_d_conv1 = tf.get_variable('wd1', [5, 5, 16, 32], initializer=w_initializer)
b_d_conv1 = tf.get_variable('bd1', [1, ], initializer=b_initializer)
output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 7, 7, 16])
h_d_conv1 = tf.nn.relu(deconv2d(code_layer, W_d_conv1, output_shape_d_conv1))
output_shape_d_pool1 = tf.stack([tf.shape(x)[0], 14, 14, 16])
h_d_pool1 = max_unpool_2x2(h_d_conv1, output_shape_d_pool1)

W_d_conv2 = tf.get_variable('wd2', [5,5,1,16], initializer=w_initializer)
b_d_conv2 = tf.get_variable('bd2', [16, ], initializer=b_initializer)
output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 14, 14, 1])
h_d_conv2 = tf.nn.relu(deconv2d(h_d_pool1, W_d_conv2, output_shape_d_conv2))
output_shape_d_pool2 = tf.stack([tf.shape(x)[0], 28, 28, 1])
h_d_pool2 = max_unpool_2x2(h_d_conv2, output_shape_d_pool2)
x_reconstruct = h_d_pool2
print("reconstruct layer shape : %s" % x_reconstruct.get_shape())


cost = tf.reduce_mean(tf.pow(x_reconstruct - x_origin, 2))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

sess = tf.InteractiveSession()
batch_size = 60
training_epochs = 1
display_step = 1
examples_to_show = 10
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
            # batch_xs = batch_xs[0].reshape([-1, 28, 28, 1])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    encode_decode = sess.run(
        x_reconstruct, feed_dict={x: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()