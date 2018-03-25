# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Using Placeholders
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph)

print(sess.run(y, feed_dict={x: rand_array}))