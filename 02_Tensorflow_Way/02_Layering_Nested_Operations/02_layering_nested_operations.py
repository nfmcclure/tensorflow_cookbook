# Layering Nested Operations
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
my_array = np.array([[1., 3., 5., 7., 9.],
                   [-2., 0., 2., 4., 6.],
                   [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 1st Operation Layer = Multiplication
prod1 = tf.matmul(x_data, m1)

# 2nd Operation Layer = Multiplication
prod2 = tf.matmul(prod1, m2)

# 3rd Operation Layer = Addition
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

merged = tf.merge_all_summaries()

if not os.path.exists('tensorboard_logs/')
    os.makedirs('tensorboard_logs/')

my_writer = tf.train.SummaryWriter('tensorboard_logs/', sess.graph)
