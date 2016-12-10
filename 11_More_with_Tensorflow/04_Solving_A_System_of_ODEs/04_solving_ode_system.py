# -*- coding: utf-8 -*-
# Solving a Sytem of ODEs
#----------------------------------
#
# In this script, we use Tensorflow to solve a sytem
#   of ODEs.
#
# The system of ODEs we will solve is the Lotka-Volterra
#   predator-prey system.


# Declaring Operations
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Open interactive graph session
sess = tf.Session()

# Discrete Lotka-Volterra predator/prey equations
#
# X(t+1) = X(t) + (aX(t) + bX(t)Y(t)) * t_delta # Prey
#
# Y(t+1) = Y(t) + (cY(t) + dX(t)Y(t)) * t_delta # Predator

# Declare constants and variables
x_initial = tf.constant(1.0)
y_initial = tf.constant(1.0)
X_t1 = tf.Variable(x_initial)
Y_t1 = tf.Variable(y_initial)

# Make the placeholders
t_delta = tf.placeholder(tf.float32, shape=())
a = tf.placeholder(tf.float32, shape=())
b = tf.placeholder(tf.float32, shape=())
c = tf.placeholder(tf.float32, shape=())
d = tf.placeholder(tf.float32, shape=())

# Discretized ODE update
X_t2 = X_t1 + (a * X_t1 + b * X_t1 * Y_t1) * t_delta
Y_t2 = Y_t1 + (c * Y_t1 + d * X_t1 * Y_t1) * t_delta

# Update to New Population
step = tf.group(
  X_t1.assign(X_t2),
  Y_t1.assign(Y_t2))
  
init = tf.initialize_all_variables()
sess.run(init)

# Run the ODE
prey_values = []
predator_values = []
for i in range(1000):
    # Step simulation (using constants for a known cyclic solution)
    step.run({a: (2./3.), b: (-4./3.), c: -1.0, d: 1.0, t_delta: 0.01}, session=sess)
    # Store each outcome
    temp_prey, temp_pred = sess.run([X_t1, Y_t1])
    prey_values.append(temp_prey)
    predator_values.append(temp_pred)

# Visualize the output
plt.plot(prey_values)
plt.plot(predator_values)
plt.legend(['Prey', 'Predator'], loc='upper right')
plt.show()