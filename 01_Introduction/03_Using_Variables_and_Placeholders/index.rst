创建变量和占位符
^^^^^^^^^^^^^^^^^

现在我们知道如何创建张量，我们可以进一步探讨如何将张量用 :literal:`Variable()` 函数打包来创建相应的变量。

我们也可以将任何 :literal:`numpy array` 转变成Python的列表，或者将常数用 :literal:`convert_to_tensor()` 转化成张量。值得注意的是， :literal:`convert_to_tensor` 也接受张量，以便我们想通过函数来计算。

区分占位符和变量是十分重要的。变量是算法的参数而TensorFlow一直都在改变这些变量来优化算法。占位符是允许你输入特定类型和大小的数据的一类对象，这类对象的结果取决于计算图的计算结果，比如计算结果的期望值。

.. code:: python
      
      >>> my_var = tf.Variable(tf.zeros([1,20]))
      >>> sess.run(my_var)
      Traceback (most recent call last):
      ...
      FailedPreconditionError: 2 root error(s) found. 


需要注意的是，直接运行 :literal:`sess.run(my_var)` 会产生一个错误。因为TensorFlows是运用计算图来运作的，我们需要对变量进行初始化才能输出结果。后面，我们可能会碰到很多
初始化操作。对于这个代码来说，我们可以调用 :literal:`my_var.initializer` 来对一个变量初始化。

.. code:: python
   
   >>> sess.run(my_var.initializer)
   >>> sess.run(my_var)
   array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

初始化是用对应的方法将变量放在计算图上。这里有一简单初始化的实例::
   
   >>> my_var1 = tf.Variable(tf.zeros([2,3]))
   >>> sess = tf.compat.v1.Session()
   # 初始化全局变量
   >>> initialize_op = tf.compat.v1.global_variables_initializer()
   >>> sess.run(initialize_op)

可以通过Tensorboard来查看创建并初始化变量之后的计算图。

占位符，顾名思义，就是占据一定的位置，用于在计算图中输入数据。占位符可以通过 :literal:`feed_dict` 参数来输入数据。为了将占位符放在计算图上，我们至少对占位符进行一次运算。我们初始化图谱，把 :literal:`x` 声明成一个占位符，将 :literal:`y` 定义成与 :literal:`x` 相等，也就是返回 :literal:`x` ，然后将数据传入 :literal:`x` 的占位符并运行等式操作 :code:`y=x` 。值得注意的是，TensorFlow不会返回一个在feed dictionary中自引占位符(现版本是可以返回的)。下面是举一个例子：

.. code:: python
   
      >>> import numpy as np
      >>> sess = tf.compat.v1.Session()
      >>> x = tf.compat.v1.placeholder(tf.float32,shape=[2,2])
      >>> y = tf.compat.v1.identity(x)
      >>> x_vals = np.random.rand(2,2)
      >>> sess.run(y, feed_dict={x: x_vals})
      array([[0.8200612 , 0.53398275],
       [0.5647656 , 0.84022015]], dtype=float32)
       
      >>> sess.run(x,feed_dict={x: x_vals})
      array([[0.8200612 , 0.53398275],
       [0.5647656 , 0.84022015]], dtype=float32)
      
在计算图运行的过程中，我们还需要告诉TensorFlow何时初始化我们创建的变量。TensorFlow必须知道何时何处初始化变量。尽管每个变量名都有 :code:`initializer` 方法， 但是通常情况下，最普遍的方法就是用 :code:`helper` 函数，也就是 :code:`global_variables_initializer()` 。 这个函数在计算图中创建了一个操作，让所有的变量都进行了初始化::
   
   >>> initializer_op = tf.compat.v1.global_variables_initializer()

但是如果我们想基于初始化另一个变量的结果来对我们想要创建的变量进行初始化，我们需要按照顺序进行初始化，比如：

.. code:: python
   
   >>> sess = tf.compat.v1.Session()
   >>> first_var = tf.Variable(tf.zeros([2,3]))
   >>> sess.run(first_var.initializer)
   # 取决于第一个变量
   >>> second_var = tf.Variable(tf.zeros_like(first_var))
   >>> sess.run(second_var.initializer)

创建特定的变量
^^^^^^^^^^^^^^^^^

首先，让我们通过声明行与列的大小来创建特定大小的变量吧。|emoticons1|

.. |emoticons1| unicode:: 0x1F608

.. code:: python
   
   >>> row_dim = 2
   >>> col_dim = 3

将变量初始化为0填充张量或1填充张量。

.. code:: python

   >>> zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
   >>> ones_var = tf.Variable(tf.ones([row_dim, col_dim]))
   
同样，我们也需要将变量进行初始化然后才能输出结果。

.. code:: python

   >>> sess.run(zero_var.initializer)
   >>> sess.run(ones_var.initializer)
   >>> print(sess.run(zero_var))
   [[ 0.  0.  0.]
   [ 0.  0.  0.]]
   >>> print(sess.run(ones_var))
   [[ 1.  1.  1.]
   [ 1.  1.  1.]]
   
基于其他张量的形状创建张量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果一个变量张量的形状取决于另一个变量张量，那么我们可以用TensorFlow的内置函数 :code:`ones_like()` 和 :code:`zeros_like()`

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
   
常数填充变量张量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这里我们展示一下如何创建常数填充变量张量

.. code:: python

   >>> fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
   >>> sess.run(fill_var.initializer)
   >>> print(sess.run(fill_var))
   [[-1 -1 -1]
   [-1 -1 -1]]
   
我们也可以通过一个数组或者常数列表来创建一个变量张量。

.. code:: python
   
   # 通过常数列表来创建张量
   >>> const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
   # 通过常数数组来创建变量张量
   >>> const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))
   
   >>> sess.run(const_var.initializer)
   >>> sess.run(const_fill_var.initializer)

   >>> print(sess.run(const_var))
   [8 6 7 5 3 0 9]
   >>> print(sess.run(const_fill_var))
   [[-1 -1 -1]
   [-1 -1 -1]]

   
基于序列和range来创建变量张量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们也可以通过TensorFlow中序列产生函数来创建张量。TensorFlow的函数 :code:`linspace()` 和 :code:`range()` 的运行方式和 :literal:`python` 和 :literal:`numpy` 中是一样的。

.. code:: python
   
   # TensorFlow的中linspace
   >>> linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) 
   # Generates [0.0, 0.5, 1.0] includes the end

   # TensorFlow的range
   >>> sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) 
   # Generates [6, 9, 12] doesn't include the end

   >>> sess.run(linear_var.initializer)
   >>> sess.run(sequence_var.initializer)

   >>> print(sess.run(linear_var))
   [ 0.   0.5  1. ]
   >>> print(sess.run(sequence_var))
   [6  9 12]

随机数变量张量
^^^^^^^^^^^^^^^^^^^^^

我们也可以创建随机数变量张量。

.. code:: python
   
   >>> rnorm_var = tf.compat.v1.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
   >>> runif_var = tf.compat.v1.random_uniform([row_dim, col_dim], minval=0, maxval=4)

   >>> print(sess.run(rnorm_var))
   [[ 1.1772728   1.36544371 -0.89566803]
    [-0.02099477 -0.17081328  0.2029814 ]]
   >>> print(sess.run(runif_var))
   [[ 2.54200077  1.42822504  1.34831095]
   [ 2.28473616  0.36273813  0.70220995]]
   
在TensorBoard中进行变量创建的可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了在Tensorboard中可视化变量创建的过程(第十一章有详细的介绍)，我们需要重设计算图并进行全局变量初始化操作。

.. code:: python
   
   # 重设计算图
   >>> ops.reset_default_graph()
   
   # 开始一个graph session
   >>> sess = tf.compat.v1.Session()
   
   # 创建变量张量
   >>> my_var = tf.Variable(tf.zeros([1,20]))

   # 将summary加到Tensorboard上
   >>> merged = tf.compat.v1.summary.merge_all()

   # 初始化图形写入
   >>> writer = tf.compat.v1.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

   # 全局变量初始器
   >>> initialize_op = tf.compat.v1.global_variables_initializer()

   # 变量初始化
   >>> sess.run(initialize_op)
   
下面，我们就可以在CLI(Commmand-Line-Interface)写入：

.. code:: bash
   
   $ tensorboard --logdir=/tmp

它会告诉我们网页链接，去查看Tensorboard。默认的值为: http://localhost:6006/

.. image:: /01_Introduction/images/02_variable.png

在这张图上，我们可以看到只有一个变量，这个变量初始化成零张量。灰色的区域是操作符和涉及到的常数的详细图解。右上角是省略的计算图。如果想要了解更多关于计算图的知识，请参考第十章第一部分。

下载本节 :download:`Jupyter Notebook </01_Introduction/02_Creating_and_Using_Tensors/tensorflow2.1tutorialch1sec2&3.ipynb>`

.. image:: /01_Introduction/images/03_placeholder.png
