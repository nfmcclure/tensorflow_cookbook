.. important::

   张量是TensorFlow在计算图上用于处理的主要数据源。我们可以把这些张量声明为变量，并将它们像占位符一样导入。首先，我们必须知道如何创建张量。
   
.. attention::
   
   但我们创建一个张量，然后声明它为变量之后，TensorFlow在计算图中创建出了多个图结构。值得注意的是通过创建张量，TensorFlow并没有在计算图增加任何东西。我们下一节会讲到这点。
   
这一节主要讲解在TensorFlow中创建张量的方法。首先，我们开始加载TensorFlow并开始重设计算图。

.. code:: python
   
    >>> import tensorflow as tf
    >>> from tensorflow.python.framework import ops
    >>> ops.reset_default_graph()

--------------

.. attention:: tensorflow.python.framework.ops.reset_default_graph模块介绍

.. automodule:: tensorflow.python.framework.ops.reset_default_graph
   :members:
   :undoc-members:
   :show-inheritance:

计算图
^^^^^^^^^^^^^^^^^^^^^

用 :literal:`tf.Session()` 开始吧！


.. code:: python
     
     >>> sess = tf.Session()
     
创建张量
^^^^^^^^^^^^^^^^^

TensorFlow有一些内置函数可以用创建变量张量。例如我们可以通过 :literal:`tf.zeros()` 来创建一个预设形状的零张量。比如：

.. code:: python
    
    >>> my_tensor = tf.zeros([1,20])
    
然后，我们可以通过 :literal:`run` 方法的调用来输出张量。

.. code:: python
    
    >>> sess.run(my_tensor)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  
    0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

---------------

.. hint:: 几种类型的张量:
   
   - 创建0填充张量::
      
      >>> import tensorflow as tf
      >>> zero_tsr = tf.zeros([row_dim, col_dim])
     |zero filled tensor|
     
   - 创建1填充张量::
      
      
      
   - 创建常数填充张量:
     |constant filled tensor|
   - 由给定的数创建一个张量：

   
.. |zero filled tensor| replace:: :literal:`[row_dim, col_dim]` 需要代入具体数字才可以输出。


TensorFlow算法需要知道哪些对象是变量哪些是常数。两个对象的区别我们在这一章中会解释，现在我们用TensorFlow的函数``tf.variable``来创建一个变量。

.. code:: python
      
      >>> my_var = tf.Variable(tf.zeros([1,20]))


Note that you can not run `sess.run(my_var)`, this would result in an error. Because TensorFlow operates with computational graphs, we have to create a variable intialization operation in order to evaluate variables. We will see more of this later on. For this script, we can initialize one variable at a time by calling the variable method my_var.initializer.

.. code:: python
   
   >>> sess.run(my_var.initializer)
   >>> sess.run(my_var)
   array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)
         
Let's first start by creating variables of specific shape by declaring our row and column size.

.. code:: python
   
   >>> row_dim = 2
   >>> col_dim = 3
   
Here are variables initialized to contain all zeros or ones.

.. code:: python

   >>> zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
   >>> ones_var = tf.Variable(tf.ones([row_dim, col_dim]))
   
Again, we can call the initializer method on our variables and run them to evaluate thier contents.

.. code:: python

   >>> sess.run(zero_var.initializer)
   >>> sess.run(ones_var.initializer)
   >>> print(sess.run(zero_var))
   [[ 0.  0.  0.]
   [ 0.  0.  0.]]
   >>> print(sess.run(ones_var))
   [[ 1.  1.  1.]
   [ 1.  1.  1.]]
   
Creating Tensors Based on Other Tensor's Shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the shape of a tensor depends on the shape of another tensor, then we can use the TensorFlow built-in functions `ones_like()` or `zeros_like()`.

.. code:: python

   >>> zero_similar = tf.Variable(tf.zeros_like(zero_var))
   >>> ones_similar = tf.Variable(tf.ones_like(ones_var))
   >>> sess.run(ones_similar.initializer)
   >>> sess.run(zero_similar.initializer)
   >>> print(sess.run(ones_similar))
   [[ 1.  1.  1.]
   [ 1.  1.  1.]] 
   >>> print(sess.run(zero_similar))
   [[ 0.  0.  0.]
   [ 0.  0.  0.]]
   
Filling a Tensor with a Constant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is how we fill a tensor with a constant.

.. code:: python

   >>> fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
   >>> sess.run(fill_var.initializer)
   >>> print(sess.run(fill_var))
   [[-1 -1 -1]
   [-1 -1 -1]]
      
We can also create a variable from an array or list of constants.

.. code:: python
   
   # Create a variable from a constant
   >>> const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
   # This can also be used to fill an array:
   >>> const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))
   
   >>> sess.run(const_var.initializer)
   >>> sess.run(const_fill_var.initializer)

   >>> print(sess.run(const_var))
   [8 6 7 5 3 0 9]
   >>> print(sess.run(const_fill_var))
   [[-1 -1 -1]
   [-1 -1 -1]]

   
Creating Tensors Based on Sequences and Ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also create tensors from sequence generation functions in TensorFlow. The TensorFlow function `linspace()` and `range()` operate very similar to the python/numpy equivalents.

.. code:: python
   
   # Linspace in TensorFlow
   >>> linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) 
   # Generates [0.0, 0.5, 1.0] includes the end

   # Range in TensorFlow
   >>> sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) 
   # Generates [6, 9, 12] doesn't include the end

   >>> sess.run(linear_var.initializer)
   >>> sess.run(sequence_var.initializer)

   >>> print(sess.run(linear_var))
   [ 0.   0.5  1. ]
   >>> print(sess.run(sequence_var))
   [6  9 12]

Random Number Tensors
^^^^^^^^^^^^^^^^^^^^^
We can also initialize tensors that come from random numbers like the following.

.. code:: python
   
   >>> rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
   >>> runif_var = tf.random_uniform([row_dim, col_dim], minval=0, maxval=4)

   >>> print(sess.run(rnorm_var))
   [[ 1.1772728   1.36544371 -0.89566803]
    [-0.02099477 -0.17081328  0.2029814 ]]
   >>> print(sess.run(runif_var))
   [[ 2.54200077  1.42822504  1.34831095]
   [ 2.28473616  0.36273813  0.70220995]]
   
Visualizing the Variable Creation in TensorBoard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To visualize the creation of variables in Tensorboard (covered in more detail in Chapter 11), we will reset the computational graph and create a global initializing operation.

.. code:: python
   
   # Reset graph
   >>> ops.reset_default_graph()

   # Start a graph session
   >>> sess = tf.Session()

   # Create variable
   >>> my_var = tf.Variable(tf.zeros([1,20]))

   # Add summaries to tensorboard
   >>> merged = tf.summary.merge_all()

   # Initialize graph writer:
   >>> writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

   # Initialize operation
   >>> initialize_op = tf.global_variables_initializer()

   # Run initialization of variable
   >>> sess.run(initialize_op)
   
We now run the following command in our command prompt:

.. code:: bash
   
   $ tensorboard --logdir=/tmp

And it will tell us the URL we can navigate our browser to to see Tensorboard. The default should be: http://0.0.0.0:6006/

.. image:: /01_Introduction/images/02_variable.png


