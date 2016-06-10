# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in Tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Using Placeholders
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/variable_logs", sess.graph_def)

print(sess.run(y, feed_dict={x: rand_array}))