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
     
     # 适用于低版本Tensorflow运行
     >>> sess = tf.Session()
     # 适用于2.0版本TensorFlow运行, 由于版本不同，必须先运行下面的命令run才能工作
     >>> tf.compat.v1.disable_eager_execution()
     # compat指的是兼容v1版本的Tensorflow
     >>> sess = tf.compat.v1.Session()
     
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
      >>> row_dim, col_dim = 3, 5
      >>> zero_tsr = tf.zeros([row_dim, col_dim])
      >>> sess.run(zero_tsr)
      array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]], dtype=float32)
    |zero filled tensor|
   - 创建1填充张量::
      
      >>> import tensorflow as tf
      >>> row_dim, col_dim = 6, 7
      >>> ones_tsr = tf.ones([row_dim, col_dim])
      >>> sess.run(ones_tsr)
      array([[1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1.]], dtype=float32)
    |one filled tensor|  
   - 创建常数填充张量::
      
      >>> import tensorflow as tf
      >>> row_dim, col_dim = 6, 7
      >>> filled_tsr = tf.fill([row_dim, col_dim],42)
      >>> sess.run(filled_tsr)
      array([[42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42],
       [42, 42, 42, 42, 42, 42, 42]], dtype=int32)
    |constant filled tensor|
   - 由给定的数创建一个张量::
      
      >>> import tensorflow as tf
      >>> constant1_tsr = tf.constant([1,2,3])
      >>> sess.run(constant1_tsr)
      [1 2 3]
      >>> constant2_tsr = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
      >>> sess.run(constant2_tsr)
      [[1 2 3]
       [4 5 6]
       [7 8 9]]
    |existing tensor|
   - 创建相似类型的张量::
      
      >>> zeros_similar = tf.zeros_like(constant1_tsr)
      >>> sess.run(zeros_similar)
      array([0, 0, 0], dtype=int32)
      >>> ones_similar = tf.ones_like(constant2_tsr)
      >>> sess.run(ones_similar)
      array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]], dtype=int32)
   - 创建序列张量::
      
      # linspace必须规定start的数是bfloat16, float16, float32, float64当中的一种
      >>> linear_tsr = tf.linspace(start=0.0,stop=100,num=11)
      # stop=100，最后一位数包括100
      >>> sess.run(linear_tsr)
      array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.],
      dtype=float32)
      
      # range的start比较宽松，可以是整数。
      >>> integer_seq_tsr = tf.range(start=6,limit=15,delta=3)
      # limit=15, 最后一位数不包括15
      >>> sess.run(integer_seq_tsr)
      array([ 6,  9, 12], dtype=int32)
   
   - 创建随机张量::
      
      # 下面创建一个符合均匀分布(uniform distribution)的随机数张量
      >>> row_dim, col_dim = 8, 8
      # 包含minval,不包含maxval
      >>> randuif_tsr = tf.compat.v1.random_uniform([row_dim, col_dim], minval=0, maxval=1)
      >>> sess.run(randuif_tsr)
      array([[0.67701995, 0.18257272, 0.57032907, 0.36612427, 0.9630263 ,
        0.95663846, 0.8787807 , 0.17861104],
       [0.4416871 , 0.9086859 , 0.3647703 , 0.21749687, 0.45980632,
        0.36322677, 0.45077944, 0.18235803],
       [0.23256958, 0.7551502 , 0.574257  , 0.31542778, 0.47067642,
        0.59856176, 0.7479335 , 0.9510181 ],
       [0.7199836 , 0.96217847, 0.6937009 , 0.7456448 , 0.24289751,
        0.85406077, 0.6463398 , 0.25423837],
       [0.95849264, 0.6280341 , 0.5537604 , 0.49765468, 0.07170725,
        0.19740784, 0.6923628 , 0.6402495 ],
       [0.93710315, 0.7305033 , 0.96696365, 0.46475697, 0.06905127,
        0.7408395 , 0.712886  , 0.00653875],
       [0.5427816 , 0.22150195, 0.460876  , 0.35927665, 0.32854652,
        0.13955867, 0.56905234, 0.97424316],
       [0.05879259, 0.3620267 , 0.81892705, 0.08734441, 0.361081  ,
        0.6088749 , 0.3457687 , 0.69742644]], dtype=float32)
     
      # 创建一个符合正态分布(normal distribution)的随机数张量
      >>> row_dim, col_dim = 8, 8
      >>> randnorm_tsr = tf.compat.v1.random_normal([row_dim,col_dim], mean=0.0, stddev=1.0)
      >>> sess.run(randnorm_tsr)
      array([[-1.3551812 ,  0.44311747, -0.07009585, -0.3532377 , -0.182757, 0.13516597,  0.4071887 ,  0.27975908],
             [ 0.42585635,  0.5364396 , -0.6653683 ,  0.35444063, -1.0977732 , -0.59936076, -0.36046746, -0.07343452],
             [-0.919484  ,  0.39717674,  0.7935889 , -0.9890499 , -1.133034  , 1.0666726 , -0.968096  ,  1.2872337 ],
             [-0.66985756, -1.1499914 ,  1.7560692 , -0.10894807,  1.1841142 , 0.22291774, -0.951817  , -0.44093087],
             [-1.0684127 , -1.0498457 ,  2.9362292 , -2.013448  ,  0.4025221 , -1.1769909 , -0.05197304, -1.4978093 ],
             [-0.38958997,  0.39442828,  0.97004807,  0.13250023, -1.2196823 , 0.70165646, -1.0563769 ,  0.10399553],
             [ 0.41292164, -0.03876609, -1.2176208 ,  0.8764762 , -0.31439155, 0.06191747, -0.87645555,  0.5363252 ],
             [-1.112473  ,  2.0940979 ,  1.3212632 , -0.14039427,  1.903088  , -1.0271009 ,  0.9657831 , -0.8105811 ]], dtype=float32)
     
      # 创建有界限的正态分布的随机数张量。即截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
      >>> row_dim, col_dim = 8, 8
      >>> runcnomr_tsr = tf.compat.v1.truncated_normal([row_dim,col_dim],mean=0.0, stddev=1.0)
      >>> sess.run(runcnomr_tsr)
      array([[ 0.57215023, -0.02053498,  0.06714377, -1.2676795 ,  0.33678156, 0.803336  , -0.10746168, -1.073573  ],
             [-1.6188551 ,  0.26903188, -0.94024265, -1.0895174 , -0.3667447 , -1.934491  ,  0.16837268,  0.14565438],
             [ 1.3880031 ,  0.25730732, -1.2500429 ,  1.2005805 , -0.6324095 , -0.5305861 , -0.86797935,  0.58874166],
             [-0.34581357, -0.69425064, -1.8915199 , -0.7588796 , -0.4680857 , -0.6425717 , -0.35572565,  0.33899295],
             [-0.50731635, -1.191694  ,  1.2362499 , -1.6300774 , -1.7100778 , -0.5509973 ,  1.7180538 , -0.05677445],
             [-0.6379802 ,  1.0952779 , -0.57122874,  0.35372928,  0.99445486, -0.37966916, -1.5172375 , -0.2665035 ],
             [ 1.631818  ,  0.79803437,  1.6253722 , -0.02572301, -0.1393287 , -1.8196368 ,  0.03887375,  0.5125945 ],
             [ 1.0057242 , -0.93407774, -0.06123861, -0.16788454,  0.62762713, 1.2990429 ,  0.5621885 ,  0.6616505 ]], dtype=float32)
     
      # 张量乱序化, runcnorm_tsr只是一个张量例子
      >>> shuffled_output = tf.compat.v1.random_shuffle(runcnomr_tsr)
      >>> sess.run(shuffled_output)
      array([[-1.5790983 ,  1.390395  , -1.5734539 , -1.2803887 , -0.36437657, 0.30741617,  0.9532189 ,  0.43124342],
             [-0.21545868,  0.5560213 , -1.1023369 , -1.365619  , -1.1592077 , 1.516915  , -0.386228  ,  1.6577938 ],
             [-0.56759614, -1.7026372 , -0.39424533,  0.20800175, -0.49035162, -1.4874234 ,  0.5077964 , -0.97859126],
             [-1.657173  , -1.2724566 , -0.12424537, -0.09589671, -1.3740199 , -0.19883458, -0.24118501, -0.25363442],
             [ 1.1784359 ,  1.6380433 ,  0.22968899, -0.3419656 , -0.5073284 , -0.37669885, -0.00905402,  0.10761048],
             [ 0.94037515,  0.14280881,  0.44833976, -0.3870774 ,  0.5403837 , -0.96695757,  0.54265535, -0.56348246],
             [ 0.8507602 , -1.2580659 ,  1.1683265 ,  1.4664146 ,  0.59427595, -0.49156505, -1.1784973 ,  0.14118564],
             [ 0.2539443 , -1.3915894 , -0.6779825 , -0.66317   ,  0.01306346, 0.5949122 , -1.409377  , -0.38872847]], dtype=float32)
     
      # 张量裁剪，第二个参数cropped_size必须是[n,m]格式
      >>> cropped_output = tf.compat.v1.random_crop(runcnomr_tsr,[4,4])
      >>> sess.run(cropped_output)
      array([[-0.2630262 ,  1.2543985 ,  0.14447008, -0.00760976],
             [-1.2469869 , -0.3482599 ,  1.4325598 ,  0.03993478],
             [-1.7399155 ,  1.0116926 , -0.22996971,  1.4531476 ],
             [-0.01253414, -1.0832093 , -1.2577766 ,  1.4000101 ]],dtype=float32)
    
      # 张量乱序化和裁剪操作都不是原处改变(in-place changes), 但是每次运行sess.run, 得到随机张量都会不一样, 必要的时候需要赋值语句
      >>> sess.run(runcnomr_tsr)
      array([[-0.01128286,  0.10473254,  0.7416311 ,  0.12495294, -0.621709  , 0.08294442, -0.3259678 ,  1.9100105 ],
             [-0.7485761 ,  1.871997  ,  0.3522917 , -0.27935842, -0.14542657, -0.06015118,  0.02190878, -0.07216269],
             [ 0.17552952,  0.395008  ,  0.06362368,  0.09165095,  0.41191736, 0.4416554 ,  0.5326085 ,  0.19600478],
             [ 1.1290088 ,  1.6767063 , -0.06439265,  0.68743473, -0.76912147, -0.74357826, -0.62004423, -1.5831621 ],
             [ 0.24502024, -0.04311023,  0.36677885, -0.7533206 , -0.83164   , 1.3448423 ,  0.8730749 , -0.13600092],
             [ 0.12533237,  0.49264213,  0.48348406, -0.03921305,  1.0805569 , 0.8118515 ,  0.6512441 , -0.11669531],
             [ 0.72900176,  1.8130132 ,  1.3789786 ,  0.519455  , -1.179993  , -1.0784473 ,  1.1946204 , -1.0734705 ],
             [ 0.68626446,  1.2634999 , -0.03061075, -1.3075253 , -0.4238513 , -1.4350135 ,  0.70656526,  1.2966055 ]], dtype=float32)
      >>> my_tsr = sess.run(runcnomr_tsr)
     
      # 后面我们会谈到图像处理，可能会用到下面的代码
      >>> import matplotlib.pyplot as plt
      >>> %matplotlib inline
      >>> image_raw_data_jpg=tf.compat.v1.gfile.GFile("yourimage.jpg","rb").read()
      >>> with sess as session:
      ...    img_data=tf.image.decode_jpeg(image_raw_data_jpg)
      ...    plt.figure(1)
      ...    print(session.run(img_data))
      ...    plt.imshow(img_data.eval())
      [[[249 253 254]
        [249 253 254]
        [249 253 254]
        ...
        [255 255 255]
        [255 255 255]
        [255 255 255]]

      [[249 253 254]
       [249 253 254]
       [249 253 254]
       ...
       ...
    
      # 运行了with sess as session之后, session会关闭，此时需要重新打开
      >>> sess = tf.compat.v1.Session()
      >>> cropped_image = tf.compat.v1.random_crop(img_data, [3, 1, 3])
      >>> sess.run(cropped_image)
      array([[[255, 255, 255]],

             [[255, 255, 255]],

             [[255, 255, 255]]], dtype=uint8)
     
.. |zero filled tensor| replace:: :literal:`[row_dim, col_dim]` row_dim是行维度，col_dim是列维度，需要代入具体数字才可以输出。
.. |one filled tensor| replace:: :literal:`[row_dim, col_dim]` row_dim是行维度，col_dim是列维度，同样需要代入具体数字才可以输出。
.. |constant filled tensor| replace:: :literal:`[row_dim, col_dim]` row_dim是行维度，col_dim是列维度，同样需要代入具体数字才可以输出。
.. |existing tensor| replace:: :literal:`tf.constant([...])` 可以改变输入常数的维度来输出对应的维度的常数张量。
.. |runcnorm_tsr| replace:: :literal:`runcnorm_tsr`
.. |cropped_size| replace:: :literal:`cropped_size`
.. |[n,m]| replace:: :literal:`[n,m]`
.. |sess.run| replace:: :literal:`sess.run`
.. |with sess as session| replace:: :literal:`with sess as session`
.. |session| replace:: :literal:`session`

创建变量
^^^^^^^^^^^^^^^^^

现在我们知道如何创建张量，我们可以进一步探讨如何将张量用 :literal:`Variable()`函数打包来创建相应的变量。

.. code:: python
      
      >>> my_var = tf.Variable(tf.zeros([1,20]))
      >>> sess.run(my_var)
      FailedPreconditionError: 2 root error(s) found.
      
需要注意的是，直接运行 :literal:`sess.run(my_var)` 会产生一个错误。因为TensorFlows是运用计算图来运作的，我们需要对变量进行初始化才能输出结果。后面，我们可能会碰到很多
初始化操作。对于这个代码来说，我们可以调用 :literal:`my_var.initializer` 来对一个变量初始化。

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


