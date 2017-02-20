# Implementing Different Layers
#---------------------------------------
#
# We will illustrate how to use different types
# of layers in TensorFlow
#
# The layers of interest are:
#  (1) Convolutional Layer
#  (2) Activation Layer
#  (3) Max-Pool Layer
#  (4) Fully Connected Layer
#
# We will generate two different data sets for this
#  script, a 1-D data set (row of data) and
#  a 2-D data set (similar to picture)

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import random
from tensorflow.python.framework import ops
ops.reset_default_graph()

#---------------------------------------------------|
#-------------------1D-data-------------------------|
#---------------------------------------------------|

# Create graph session 
sess = tf.Session()

# Generate 1D data
data_size = 25
data_1d = np.random.normal(size=data_size)

# Placeholder
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

#--------Convolution--------
def conv_layer_1d(input_1d, my_filter):
    # TensorFlow's 'conv2d()' function only works with 4D arrays:
    # [batch#, width, height, channels], we have 1 batch, and
    # width = 1, but height = the length of the input, and 1 channel.
    # So next we create the 4D array by inserting dimension 1's.
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution with stride = 1, if we wanted to increase the stride,
    # to say '2', then strides=[1,1,2,1]
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding="VALID")
    # Get rid of extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return(conv_output_1d)

# Create filter for convolution.
my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
# Create convolution layer
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

#--------Activation--------
def activation(input_1d):
    return(tf.nn.relu(input_1d))

# Create activation layer
my_activation_output = activation(my_convolution_output)

#--------Max Pool--------
def max_pool(input_1d, width):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, width=1, height=num_input, channels=1]
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 1, 2, 1]
    # We will also need to specify the width of the max-window ('width')
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    # Get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return(pool_output_1d)

my_maxpool_output = max_pool(my_activation_output, width=5)

#--------Fully Connected--------
def fully_connected(input_layer, num_outputs):
    # First we find the needed shape of the multiplication weight matrix:
    # The dimension will be (length of input) by (num_outputs)
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer),[num_outputs]]))
    # Initialize such weight
    weight = tf.random_normal(weight_shape, stddev=0.1)
    # Initialize the bias
    bias = tf.random_normal(shape=[num_outputs])
    # Make the 1D input array into a 2D array for matrix multiplication
    input_layer_2d = tf.expand_dims(input_layer, 0)
    # Perform the matrix multiplication and add the bias
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    # Get rid of extra dimensions
    full_output_1d = tf.squeeze(full_output)
    return(full_output_1d)

my_full_output = fully_connected(my_maxpool_output, 5)

# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}

# Convolution Output
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print(sess.run(my_full_output, feed_dict=feed_dict))

#---------------------------------------------------|
#-------------------2D-data-------------------------|
#---------------------------------------------------|

# Reset Graph
ops.reset_default_graph()
sess = tf.Session()

#Generate 2D data
data_size = [10,10]
data_2d = np.random.normal(size=data_size)

#--------Placeholder--------
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

# Convolution
def conv_layer_2d(input_2d, my_filter):
    # TensorFlow's 'conv2d()' function only works with 4D arrays:
    # [batch#, width, height, channels], we have 1 batch, and
    # 1 channel, but we do have width AND height this time.
    # So next we create the 4D array by inserting dimension 1's.
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Note the stride difference below!
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,2,2,1], padding="VALID")
    # Get rid of unnecessary dimensions
    conv_output_2d = tf.squeeze(convolution_output)
    return(conv_output_2d)

# Create Convolutional Filter
my_filter = tf.Variable(tf.random_normal(shape=[2,2,1,1]))
# Create Convolutional Layer
my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

#--------Activation--------
def activation(input_1d):
    return(tf.nn.relu(input_1d))

# Create Activation Layer
my_activation_output = activation(my_convolution_output)

#--------Max Pool--------
def max_pool(input_2d, width, height):
    # Just like 'conv2d()' above, max_pool() works with 4D arrays.
    # [batch_size=1, width=given, height=given, channels=1]
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pooling with strides = [1,1,1,1]
    # If we wanted to increase the stride on our data dimension, say by
    # a factor of '2', we put strides = [1, 2, 2, 1]
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    # Get rid of unnecessary dimensions
    pool_output_2d = tf.squeeze(pool_output)
    return(pool_output_2d)

# Create Max-Pool Layer
my_maxpool_output = max_pool(my_activation_output, width=2, height=2)


#--------Fully Connected--------
def fully_connected(input_layer, num_outputs):
    # In order to connect our whole W byH 2d array, we first flatten it out to
    # a W times H 1D array.
    flat_input = tf.reshape(input_layer, [-1])
    # We then find out how long it is, and create an array for the shape of
    # the multiplication weight = (WxH) by (num_outputs)
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input),[num_outputs]]))
    # Initialize the weight
    weight = tf.random_normal(weight_shape, stddev=0.1)
    # Initialize the bias
    bias = tf.random_normal(shape=[num_outputs])
    # Now make the flat 1D array into a 2D array for multiplication
    input_2d = tf.expand_dims(flat_input, 0)
    # Multiply and add the bias
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    # Get rid of extra dimension
    full_output_2d = tf.squeeze(full_output)
    return(full_output_2d)

# Create Fully Connected Layer
my_full_output = fully_connected(my_maxpool_output, 5)

# Run graph
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

# Convolution Output
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print(sess.run(my_activation_output, feed_dict=feed_dict))

# Max Pool Output
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print(sess.run(my_full_output, feed_dict=feed_dict))