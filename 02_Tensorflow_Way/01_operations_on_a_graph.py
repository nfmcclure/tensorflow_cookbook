# Operations on a Computational Graph
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m = tf.constant(3.)

# Multiplication
prod = tf.mul(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))

merged = tf.merge_all_summaries()
my_writer = tf.train.SummaryWriter('/home/nick/OneDrive/Documents/tensor_flow_book/Code/2_Tensorflow_Way', sess.graph)