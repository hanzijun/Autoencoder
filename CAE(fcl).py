# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Parameters

class autoencoder():
    def __init__(
            self,
            n_input = 28,
            n_step = 28,
            batch_size = 128,
            learning_rate = 0.01,
            training_epochs = 20,
            display_step = 1,
            examplesToShow = 10,
            param_file = True,
            is_train = False
                 ):
# Network Parameters
        self.batch_size = batch_size
        self.lr = learning_rate
        self.training_epochs = training_epochs
        self.display = display_step
        self.examples = examplesToShow
        self.encoder = None
        self.buildautoencoder()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./params_fcl/CAE_.ckpt")
                print("loading nerou-network params...")
            self.learn()
            self.show()
            # self.fine_tuning()
        else:
            self.fine_tuning()

    def buildautoencoder(self):
            """
            build the autoencoder containing two parts, namely encoder which is adopted to compress the origin input image or image-like data set (denote X) into x'
            and decoder which is used to reconstruct the input, x'  to X. The loss between the  X and x' drives the parameters updating.
            :return:  the reconstructed input x'
            """
            self.x = tf.placeholder(tf.float32, shape = [None, 784], name='image_origin')
            self.input = tf.reshape(self.x, [-1, 28, 28, 1])
            self.input_reconstruct = tf.placeholder(tf.float32, shape = [None, 28,28,1], name='image_reconstruct')

            with tf.variable_scope('encoder'):
                c_names, n_l1 = ['encoder_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256
                w_initializer = tf.random_normal_initializer(0., 0.1)
                b_initializer = tf.constant_initializer(0.)

                with tf.variable_scope('conv_l1'):
                    self.we_conv1 = tf.get_variable('w_encoder1', [5, 5, 1, 16], initializer=w_initializer, collections=c_names)
                    self.b_conv1 = tf.get_variable('b_encoder1', [16, ], initializer=b_initializer, collections=c_names)
                    self.conv1 = tf.nn.relu(tf.add(self.conv2d(self.input, self.we_conv1), self.b_conv1))

                with tf.variable_scope('conv_l2'):
                    self.we_conv2 = tf.get_variable('w_encoder2', [5, 5, 16, 32], initializer=w_initializer, collections=c_names)
                    self.b_conv2 = tf.get_variable('b_encoder2', [32, ], initializer=b_initializer, collections=c_names)
                    conv2 = tf.nn.relu(tf.add(self.conv2d(self.conv1, self.we_conv2), self.b_conv2))

                with tf.variable_scope('fcl_l1'):
                    conv2_reshape = tf.reshape(conv2, [-1, 7 * 7 * 32])
                    self.w_fc1 = tf.get_variable('wfc1', [7 * 7 * 32, 1024], initializer=w_initializer)
                    self.b_fc1 = tf.get_variable('bfc1', [1024], initializer=b_initializer)
                    result = tf.nn.relu(tf.matmul(conv2_reshape, self.w_fc1) + self.b_fc1)
                    self.encoder = result
                    print (self.encoder.shape)

            with tf.variable_scope('decoder'):
                w_initializer = tf.random_normal_initializer(0., 0.1)
                b_initializer = tf.constant_initializer(0)

                with tf.variable_scope('fcl_trans'):
                    self.w_fc2 = tf.get_variable('wfc2', [1024, 7 * 7 * 32], initializer=w_initializer)
                    self.b_fc2 = tf.get_variable('bfc2', [7*7*32], initializer=b_initializer)
                    result_reconstruct = tf.nn.relu(tf.matmul(self.encoder, self.w_fc2) + self.b_fc2)

                with tf.variable_scope('deconv_l1'):
                    result_reconstruct = tf.reshape(result_reconstruct, [-1, 7, 7, 32], name='result_reconstruct_3D')
                    self.wd_conv1 = tf.get_variable('w_decoder1', [5, 5, 16, 32], initializer=w_initializer)
                    b_conv1 = tf.get_variable('b_decoder1', [1, ], initializer=b_initializer)
                    output_shape_d_conv1 = tf.stack([tf.shape(self.x)[0], 14, 14, 16])
                    h_d_conv1 = tf.nn.relu(self.deconv2d(result_reconstruct, self.wd_conv1, output_shape_d_conv1))

                with tf.variable_scope('deconv_l2'):
                    self.wd_conv2 = tf.get_variable('w_decoder2', [5, 5, 1, 16], initializer=w_initializer)
                    b_conv2 = tf.get_variable('b_decoder2', [16, ], initializer=b_initializer)
                    output_shape_d_conv2 = tf.stack([tf.shape(self.x)[0], 28, 28, 1])
                    h_d_conv2 = tf.nn.relu(self.deconv2d(h_d_conv1, self.wd_conv2, output_shape_d_conv2))
                    self.input_reconstruct = h_d_conv2

                with tf.variable_scope('loss'):
                    alpha, beta, rho = 5e-6, 7.5e-6, 0.08
                    Wset = [self.we_conv1, self.we_conv2, self.w_fc1, self.w_fc2,self.wd_conv1, self.wd_conv2]
                    results = [self.conv1, self.encoder]
                    """
                    KL Divergence + L2 regularization
                    """
                    # kldiv_loss = reduce(lambda x, y: x + y, map(lambda x: tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
                    l2_loss = reduce(lambda x, y: x + y, map(lambda x: tf.nn.l2_loss(x), Wset))
                    # self.loss = tf.reduce_mean(tf.pow(self.input - self.input_reconstruct, 2))+ alpha * l2_loss + beta * kldiv_loss
                    self.loss = tf.reduce_mean(tf.pow(self.input - self.input_reconstruct, 2)) + alpha * l2_loss
                    # self.loss = tf.reduce_mean(tf.pow(self.input - self.input_reconstruct, 2))

                with tf.variable_scope('train'):
                    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self):
            """
            update the autoencoder by SGD and echo loss each epoch
            :return:
            """
            total_batch = int(mnist.train.num_examples / self.batch_size)
            # Training cycle
            for epoch in range(self.training_epochs):
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)  # max(x) = 1, min(x) = 0
                    # batch_xs = batch_xs[0].reshape([-1, 28, 28, 1])
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: batch_xs})
                # Display logs per epoch step
                if epoch % self.display == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            print("Optimization Finished!")
            self.saver.save(self.sess, "./params_fcl/CAE_.ckpt")

    def show(self):
        """
        display the performance of autoencoder
        :return: a autoencoder model using unsupervised learning
        """
        encode_decode = self.sess.run(
            self.input_reconstruct, feed_dict={self.x: mnist.test.images[:self.examples]})
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(self.examples):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()

    def fine_tuning(self):
        """
        supervised learning stage using dataset with lable
        :return:  a fine-tuned model and give the remarkable prediction
        """
        """
        Display the characteristics for image mosaicking.
        Fine-tuning phase programming for further optimizing the softmax classifier.
        """
        # image= self.sess.run(self.encoder, feed_dict={self.x: mnist.test.images[:1]})
        # image_input = tf.reshape(image, [7*7, 32])
        # plt.imshow(image_input)
        # plt.show()

        x = tf.placeholder(tf.float32, shape=[None, 784], name='image_fine-tuning')
        x_input = tf.reshape(x, [-1, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])
        w_initializer = tf.random_normal_initializer(0., 0.1)
        b_initializer = tf.constant_initializer(0)

        conv1 = tf.nn.relu(tf.add(self.conv2d(x_input, self.we_conv1), self.b_conv1))
        conv2 = tf.nn.relu(tf.add(self.conv2d(conv1, self.we_conv2), self.b_conv2))
        conv2_reshape = tf.reshape(conv2, [-1, 7 * 7 * 32])
        fcl1 = tf.nn.relu(tf.matmul(conv2_reshape, self.w_fc1) + self.b_fc1)

        w_softmax = tf.get_variable('w_softmax', [1024, 10], initializer=w_initializer)
        b_softmax = tf.get_variable('b_softmax', [10], initializer=b_initializer)
        result = tf.nn.softmax(tf.matmul(fcl1, w_softmax) + b_softmax)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=result))
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(result), reduction_indices=[1]))

        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        correct_pred = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver.restore(self.sess, "./params_fcl/CAE_.ckpt")
        print ("loading nerou-network params...")

        batch_size = 128
        total_batch = int(mnist.train.num_examples / batch_size)
        # Training cycle
        for epoch in range(5):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, acc = self.sess.run([train_op, accuracy], feed_dict={x: batch_xs, y: batch_ys,})
            if epoch % self.display == 0:
                print("accuracy","{:.9f}".format(acc))

        print("Optimization Finished!")

        image = self.sess.run(conv2_reshape, feed_dict={x: mnist.test.images[:10]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(self.examples):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(image[i], (7*7, 32)))
        plt.show()

    @staticmethod
    def conv2d( x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')
    @staticmethod
    def deconv2d(x,W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding = 'SAME')



# Building the encoder
def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho
def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

# results = [encoder_1, encoder_2, encoder_3]
# Wset = [weights['encoder_h0'], weights['encoder_h1'], weights['encoder_h2'],  tf.transpose(weights['encoder_h2']),  tf.transpose(weights['encoder_h1']), tf.transpose(weights['encoder_h0'])]
# kldiv_loss = reduce(lambda x,y : x+y, map(lambda x : tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
# l2_loss = reduce(lambda x, y : x + y, map(lambda x : tf.nn.l2_loss(x), Wset))
# Define loss and optimizer, minimize the squared error
# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + alpha * l2_loss + beta * kldiv_loss
# Launch the graph

    # # Applying encode and decode over test set

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()
if __name__ =="__main__":
    autoencoder()