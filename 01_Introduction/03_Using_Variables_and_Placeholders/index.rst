
Placeholders
-----------
We introduce how to use placeholders in TensorFlow.

First we import the necessary libraries and reset the graph session.

.. code:: python
  
  import numpy as np
  import tensorflow as tf
  from tensorflow.python.framework import ops
  ops.reset_default_graph()

Start graph session
^^^^^^^^^^^^^^^^^^^
.. code:: python
  
  sess = tf.Session()
  
Declare a placeholder
^^^^^^^^^^^^^^^^^^^^^^
We declare a placeholder by using TensorFlow's function, `tf.placeholder()`, 
which accepts a data-type argument (`tf.float32`) and a shape argument, (4,4).
Note that the shape can be a tuple or a list, `[4,4]`.

.. code:: python
  
  x = tf.placeholder(tf.float32, shape=(4, 4))
  
For illustration on how to use the placeholder, we create input data for it 
and an operation we can visualize on Tensorboard.

Note the useage of feed_dict, where we feed in the value of x into the 
computational graph.

.. code:: python
  
  # Input data to placeholder, note that 'rand_array' and 'x' are the same shape.
  rand_array = np.random.rand(4, 4)

  # Create a Tensor to perform an operation (here, y will be equal to x, a 4x4 matrix)
  y = tf.identity(x)

  # Print the output, feeding the value of x into the computational graph
  print(sess.run(y, feed_dict={x: rand_array}))
  [[ 0.1175806   0.88121527  0.00815445  0.93555111]
  [ 0.97369134  0.14595009  0.16398087  0.76570976]
  [ 0.67633879  0.11748746  0.01266815  0.32564184]
   [ 0.99007022  0.6825515   0.54524553  0.01503101]]
   
To visualize this in Tensorboard, we merge summaries and write to a log file.

.. code:: python
  
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph)
  
We run the following command at the prompt:

.. code:: bash

  tensorboard --logdir=/tmp

Which will tell us where to navigate chrome to to visualize the computational graph.
Default is http://0.0.0.0:6006/

.. image:: https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/images/03_placeholder.png
