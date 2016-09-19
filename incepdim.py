#! /usr/bin/env python2.7
import sys
sys.path.append('..')
import tensorflow as tf
from dataHandling import loadData
import numpy as np
import time
import math

fcneurons = 512
features = [16, 32]

class Model:
    def get_data(self):
        return loadData()
    
    def initialize(self):
        self.data = self.get_data()
        self.height = self.data.get_height()
        self.width = self.data.get_width()
        # Semi-Constants

        self.channels = 3
        self.classes = 3
        self.imageSize = self.height * self.width * self.channels


    def run(self, features, fcneurons, createMap=False, name='', verbose=False): 
        self.initialize()
        #input data
        x = tf.placeholder(tf.float32, [None, self.imageSize])

        #labels
        y_ = tf.placeholder(tf.float32, [None, self.classes])

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def softmax(inp, weights, bias):
            return tf.nn.softmax(tf.matmul(inp, weights) + bias)

        def fully_connected(inp, weights, bias, inWidth, inHeight, inFeat, outFeat):
            inp_reshape = tf.reshape(inp, [-1 , inHeight * inWidth * inFeat])
            return tf.nn.relu(tf.matmul(inp_reshape, weights) + bias)

        def conv_relu(inp, weight, bias):
            conv = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(conv + bias)


        # NETWORK

        x_image = tf.reshape(x, [-1, self.height, self.width, self.channels])

        #first conv layer        

        with tf.variable_scope("incep1"):

            with tf.variable_scope("conv1-0"):
                weights = weight_variable([1, 1, self.channels, 3])
                bias = bias_variable([3])
                conv1_0 = conv_relu(x_image, weights, bias)

            with tf.variable_scope("conv1-1"):
                weights = weight_variable([1, 1, self.channels, 3])
                bias = bias_variable([3])
                conv1_1 = conv_relu(x_image, weights, bias)

            with tf.variable_scope("conv1"):
                weights = weight_variable([1, 1, self.channels, features[0]])
                bias = bias_variable([features[0]])
                conv1 = conv_relu(x_image, weights, bias)

            with tf.variable_scope("conv3"):
                weights = weight_variable([3, 3, self.channels, features[0]])
                bias = bias_variable([features[0]])
                conv3 = conv_relu(conv1_0, weights, bias)

            with tf.variable_scope("conv5"):
                weights = weight_variable([5, 5, self.channels, features[0]])
                bias = bias_variable([features[0]])
                conv5 = conv_relu(conv1_1, weights, bias)

            pool = tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope("pc"):
                weights = weight_variable([1, 1, self.channels, features[0]])
                bias = bias_variable([features[0]])
                pc  = conv_relu(pool, weights, bias)

            incep1 = tf.concat(3, [conv1, conv3, conv5, pc])

        with tf.variable_scope("incep2"):


            with tf.variable_scope("conv1"):
                weights = weight_variable([1, 1, 4 * features[0], features[1]])
                bias = bias_variable([features[1]])
                conv1 = conv_relu(incep1, weights, bias)

            with tf.variable_scope("conv3"):
                weights = weight_variable([3, 3, 4 * features[0], features[1]])
                bias = bias_variable([features[1]])
                conv3 = conv_relu(incep1, weights, bias)

            with tf.variable_scope("conv5"):
                weights = weight_variable([5, 5, 4 * features[0], features[1]])
                bias = bias_variable([features[1]])
                conv5 = conv_relu(incep1, weights, bias)

            pool = tf.nn.max_pool(incep1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

            with tf.variable_scope("pc"):
                weights = weight_variable([1, 1, 4 * features[0], features[1]])
                bias = bias_variable([features[1]])
                pc  = conv_relu(pool, weights, bias)

            incep2 = tf.concat(3, [conv1, conv3, conv5, pc])


        #fully connected layer
        with tf.variable_scope("fc1"):
            weights = weight_variable([50 * 50 * (features[-1] * 4), fcneurons])
            bias = bias_variable([fcneurons])
            fc1 = fully_connected(incep2, weights, bias, 50, 50, (features[-1] * 4), fcneurons)

        #Drop out

        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        #softmax

        with tf.variable_scope("sf1"):
            weights = weight_variable([fcneurons, self.classes])
            bias = bias_variable([self.classes])
            y_conv = softmax(fc1_drop, weights, bias)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-10), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(5e-5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(5000):
                batch = self.data.get_train(50)
                if i%100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x:batch[0], y_:batch[1], keep_prob: 1.0})
                    if verbose:
                        print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            test = self.data.get_test()
            return accuracy.eval(feed_dict={x: test[0], y_: test[1], keep_prob: 1.0})



test = Model()
print test.run(features, fcneurons, verbose=True)
