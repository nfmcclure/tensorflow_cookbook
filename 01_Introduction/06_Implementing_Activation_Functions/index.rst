但我们开始使用神经网络的时候，我们会经常用到激活函数，这是因为激活函数是任何神经网络结构中不可或缺的一部分。激活函数的目的是调整权重和偏差。在TensorFlow中，激活函数是作用在张量上非线性的操作符。它们很像前面一节的数学运算符, 应用比较广泛，但是它们主要的贡献是引入计算图中非线性的计算。同样，我们需要运行一下下面的命令来创建一个 :code:`graph session` :

.. code:: python
  
  >>> import tensorflow.compat.v1 as tf
  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>> from tensorflow.python.framework import ops
  >>> ops.reset_default_graph()
  
  # 下面一行命令必须放在上面命令运行完之后，不可调换,否则容易出现empty graph
  >>> tf.disable_eager_execution()
  >>> sess = tf.Session()

激活函数都是存在于TensorFlow中神经网络(Neural Network)库中 :py:mod:`tensorflow.nn` 。除了使用内置激活函数，我们也可以使用TensorFlow运算来设计自己的激活函数。我们可以导入预先设定的函数( :code:`import tf.nn as nn` ) 或更精确一点采用 ( :code:`tf.nn` )。


线性整流函数(Rectifed Linear Unit)
-----------

.. code:: python
  
  >>> x_vals = np.linspace(start=-10,stop=10,num=100)
  >>> print(sess.run(tf.nn.relu([-3.,3.,10.])))
  [ 0.  3. 10.]
  >>> y_relu = sess.run(tf.nn.relu(x_vals))
  >>> plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
  ... plt.ylim([-5,11])
  ... plt.legend(loc='upper left')
  ... plt.show()

.. raw:: html

  <!DOCTYPE html>
  <html>
  <head><meta charset="utf-8" />
  
  <div class="output_png output_subarea ">
  <img   src="data:image/png;base64,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
  "
  >
  </div>
  
  </div>

  </div>
  </div>
  
  </div>
       </div>
    </div>
  </body>
  </html>



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
