This function introduces various operations in TensorFlow
Declaring Operations

.. code:: python

  import matplotlib.pyplot as plt
  import numpy as np
  import tensorflow as tf
  from tensorflow.python.framework import ops
  ops.reset_default_graph()

Open graph session
------------------

.. code:: python

  sess = tf.Session()
  
Arithmetic Operations
---------------------
TensorFlow has multiple types of arithmetic functions. Here we illustrate the differences
between ``div()``, ``truediv()`` and ``floordiv()``.

``div()`` : integer of division (similar to base python //)

``truediv()`` : will convert integer to floats.

``floordiv()`` : float of div()

.. code:: python

  print(sess.run(tf.div(3,4)))
  print(sess.run(tf.truediv(3,4)))
  print(sess.run(tf.floordiv(3.0,4.0)))

the output::

  0
  0.75
  0.0

Mod function:

.. code:: python

  print(sess.run(tf.mod(22.0,5.0)))

the output::

  2.0

Cross Product:

.. code:: python

  print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))

the output::

  [ 0.  0.  1.]
  
Trig functions
---------------

Sine, Cosine, and Tangent:

.. code:: python

  print(sess.run(tf.sin(3.1416)))
  print(sess.run(tf.cos(3.1416)))
  print(sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))
  
the output::

  -7.23998e-06
  -1.0
  1.0
  
  
Custom operations
------------------

Here we will create a polynomial function:

:math: 'f(x) = 3 \ast x^2-x+10'

.. code:: python

  test_nums = range(15)
  
  def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
      return(tf.subtract(3 * tf.square(x_val), x_val) + 10)

  print(sess.run(custom_polynomial(11)))

the output::
  
  362
  
What should we get with list comprehension:

.. code:: python
  
  expected_output = [3*x*x-x+10 for x in test_nums]
  print(expected_output)
  
the output::

  [10, 12, 20, 34, 54, 80, 112, 150, 194, 244, 300, 362, 430, 504, 584]
  
TensorFlow custom function output:

.. code:: python

  for num in test_nums:
      print(sess.run(custom_polynomial(num)))


the output::
  
  10
  12
  20
  34
  54
  80
  112
  150
  194
  244
  300
  362
  430
  504
  584
