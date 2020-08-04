This function introduces activation functions in TensorFlow

We start by loading the necessary libraries for this script.

.. code:: python
  
  import matplotlib.pyplot as plt
  import numpy as np
  import tensorflow as tf
  from tensorflow.python.framework import ops
  ops.reset_default_graph()
  
Start a graph session
---------------------

.. code:: python

  sess = tf.Session()
  
Initialize the X range values for plotting
-------------------------------------------

.. code:: python

  x_vals = np.linspace(start=-10., stop=10., num=100)
  
Activation Functions
--------------------
ReLU activation

.. code:: python

  print(sess.run(tf.nn.relu([-3., 3., 10.])))
  y_relu = sess.run(tf.nn.relu(x_vals))

the output::

  [  0.   3.  10.]

ReLU-6 activation

.. code:: python
  print(sess.run(tf.nn.relu6([-3., 3., 10.])))
  y_relu6 = sess.run(tf.nn.relu6(x_vals))

the output::

  [ 0.  3.  6.]

Sigmoid activation

.. code:: python
  print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
  y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

the output::

  [ 0.26894143  0.5         0.7310586 ]

Hyper Tangent activation

.. code:: python

  print(sess.run(tf.nn.tanh([-1., 0., 1.])))
  y_tanh = sess.run(tf.nn.tanh(x_vals))

the output::

  [-0.76159418  0.          0.76159418]

Softsign activation

.. code:: python

  print(sess.run(tf.nn.softsign([-1., 0., 1.])))
  y_softsign = sess.run(tf.nn.softsign(x_vals))

the output::

  [-0.5  0.   0.5]

Softplus activation

.. code:: python

  print(sess.run(tf.nn.softplus([-1., 0., 1.])))
  y_softplus = sess.run(tf.nn.softplus(x_vals))

the output::

  [ 0.31326166  0.69314718  1.31326163]

Exponential linear activation

.. code:: python

  print(sess.run(tf.nn.elu([-1., 0., 1.])))
  y_elu = sess.run(tf.nn.elu(x_vals))

the output::

  [-0.63212055  0.          1.        ]

Plot the different functions
----------------------------
.. code:: python

  plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
  plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
  plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
  plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
  plt.ylim([-1.5,7])
  plt.legend(loc='upper left')
  plt.show()

  plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
  plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
  plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
  plt.ylim([-2,2])
  plt.legend(loc='upper left')
  plt.show()



.. image:: /01_Introduction/images/06_activation_funs1.png
.. image:: /01_Introduction/images/06_activation_funs2.png
