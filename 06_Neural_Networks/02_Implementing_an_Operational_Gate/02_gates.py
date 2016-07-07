# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in Tensorflow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask Tensorflow to change the
# variable based on our loss function

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start Graph Session
sess = tf.Session()

#----------------------------------
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

multiplication = tf.mul(a, x_data)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.sub(multiplication, 50.))

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
    
#----------------------------------
# Create a nested gate:
#   f(x) = a * x + b
#
#  a --
#      |
#      |-- (multiply)--
#  x --|              |
#                     |-- (add) --> output
#                 b --|
#
#

# Start a New Graph Session
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.mul(a, x_data), b)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.sub(two_gate, 50.))

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Run loop across gate
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val, b_val = (sess.run(a), sess.run(b))
    two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))