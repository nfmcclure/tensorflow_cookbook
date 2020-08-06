引言
----

Google在2015年11月完成了对TensorFlow的开源。自从那之后，`TensorFlow <https://github.com/tensorflow/tensorflow>`_ 
已经是Github上机器学习starred最多的仓库。

为什么选择TensorFlow ? TensorFlow的受欢迎程度归因于很多方面，但是主要是因为它的计算图概念，自动微分和TensorFlow的
Python API 的架构。这些都使得程序员用TensorFlow来解决实际问题更加便捷。

Google的TensorFlow引擎有一个解决问题的独特方式。这种独特的方式使得解决机器学习问题非常有效。下面，我们会介绍TensorFlow
如何运行的基本步骤。

TensorFlow是如何运行的
-----------------------

在一开始的时候, TensorFlow中的计算可能看起来毫无必要的复杂. 但其实其中是有原因的: 也正因为TensorFlow处理计算的方式，发展
更为复杂的计算也就相对来说更为简单。这一节呢，会带领你领略一个TensorFlow算法通常工作的方式. 

现在呢，TensorFlow已经被所有的主流操作系统(Windows, Linux 和 Mac)所支持。通过这本书呢，我们只关心TensorFlow的Python库
这本书呢，会用到 `Python 3.x <https://www.python.org>`_ 和 `Tensorflow 0.12 + <https://www.tensorflow.org>`_ (我们这里会用
Python 3.7 和 TensorFlow 1.8 版本)。虽然说TensorFlow可以在CPU上运行，但是它在GPU(Graphic Processing Unit)运行得更快。
英伟达(Nvidia) Compute Capability 3.0+的显卡现在也支持TensorFlow。如果你想要在GPU上运行，你需要下载并安装 `Nvidia Cuda Toolkit <https://developer.nvidia.com/cuda-downloads>`_。 有些章节可能还依赖安装Scipy, Numpy和Scikit-learn。你可以通过下载下面的requirements.txt, 然后运行下面的命令，
来满足这些条件。

下载 :download:`requirements.txt</requirements.txt>`

.. code:: sh
      
      $ pip install -r requirements.txt 

通用TensorFlow算法概览
-------------------------------------

这里呢，我们会简单介绍一下TensorFlow算法的工作流程。大多数机器学习算法都遵循此流程。

导入或产生数据
^^^^^^^^^^^^^^^^^^^^^^^^

我们所有的机器学习算法都取决于数据。在这本书中我们要么自己产生数据，要么使用外部数据源。有时候呢，因为我们想要知道算法所
塑造的模型是否能产生期望的结果，所以有时候依赖产生的数据更好一点(因为它有参考的对象)。其他的时候呢，我们需要获取公众数据，
方法我们会在这章的第八部分提到。

转换和规范化数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有时候，数据并不是TensorFlow所能处理的正确维度。在我们使用之前，我们必须将数据进行转换。大多数算法期待的是正则化数据，
我们在这里也会用到。TensorFlow有一些内置函数可以帮助你实现数据正则化。比如：


.. code:: python
      
      # 低版本TensorFlow的用法
      >>> data = tf.nn.batch_norm_with_global_normalization(...)
      # TensorFlow 2.2的用法
      >>> data = tf.nn.batch_normalization(...)

--------------------------------------

.. attention:: tensorflow.nn.batch_normalization

.. automodule:: tensorflow.nn.batch_normalization
      :members:
      :undoc-members:
      :show-inheritance:


设置算法参数
^^^^^^^^^^^^^^^^^^^^^^^

我们使用的算法通常会有一些参数是需要我们一直保持不变的。例如，迭代次数，学习速率，或者其他的设定的参数。为了方便读者或
用户很便捷找到它们，通常将它们放在一起初始化是个很好的典范。比如：

.. code:: python
      
      >>> learning_rate = 0.01 
      >>> iterations = 1000


变量和占位符的初始化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow是需要我们告诉它，哪些是可以改变的，哪些是不可以改变的。在损失函数最小化的优化过程中，TensorFlow会改变一些变量。
为了实现这些，我们需要通过占位符(placeholders)来传入数据。变量和占位符的大小和类型都是需要我们进行初始化的，这样呢，TensorFlow
就会知道应该怎么优化。例如：

.. code:: python
      
      >>> a_var = tf.constant(42) 
      >>> x_input = tf.placeholder(tf.float32, [None, input_size]) 
      >>> y_input = tf.placeholder(tf.float32, [None, num_classes])

-------------------------

.. attention:: tensorflow.constant

.. automodule:: tensorflow.constant
      :members:
      :undoc-members:
      :show-inheritance:

----------------------

.. attention:: tensorflow.float32

.. automodule:: tensorflow.float32
      :members:
      :undoc-members:
      :show-inheritance:


定义模型结构
^^^^^^^^^^^^^^^^^^^^^^^^^^

在我们有了数据，并且将我们的变量和占位符进行初始化之后，我们就可以定义模型了。这个，我们可以通过建立一个计算图来完成。
我们告诉TensorFlow哪些操作需要在变量和占位符上完成，以实现我们的模型预测。关于计算图，我们会在第二章详细描述。
在这里我们，我们先看一下定义模型结构的例子：

.. code:: python
      
      # 低版本TensorFlow的用法
      >>> y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)
      # TensorFlow2.2的用法
      >>> y_pred = tf.add(tf.multiply(x_input, weight_matrix), b_matrix)

-----------------------------

.. attention:: tensorflow.add

.. automodule:: tensorflow.add
      :members:
      :undoc-members:
      :show-inheritance:

--------------------

.. attention:: tensorflow.multiply

.. automodule:: tensorflow.multiply
      :members:
      :undoc-members:
      :show-inheritance:

声明损失函数
^^^^^^^^^^^^^^^^^^^^^^^^^

在定义模型之后，我们就可以用TensorFlow算出结果了。这时候，我们需要定义一个损失函数。损失函数是非常重要的，因为它告诉我们
我们的预测离真实值差多少。在第二章第五节中，我们会对损失函数的类型进行详细的讲解。

.. code:: python
      
      >>> loss = tf.reduce_mean(tf.square(y_actual – y_pred))

-----------------------------

.. attention:: tensorflow.reduce_mean

.. automodule:: tensorflow.reduce_mean
      :members:
      :undoc-members:
      :show-inheritance:

-----------------------

.. attention:: tensorflow.square

.. automodule:: tensorflow.square
      :members:
      :undoc-members:
      :show-inheritance:

模型的初始化和训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

既然我们现在设置好了一切，我们可以创建一个实例或者计算图，然后通过占位符将数据传入，并通过训练让TensorFlow改变变量
来更好预测我们的训练数据。这里举出一个初始化计算图的一种方式：

.. code:: python
      
      >>> with tf.Session(graph=graph) as session:
               ...
      >>> session.run(...)
               ...
               
需要注意的是，我们也可以这样初始化计算图：

.. code:: python
      
      >>> session = tf.Session(graph=graph) 
      >>> session.run(…)


模型的评估(可选)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一旦我们建立并训练模型，我们应当通过查看它的新数据的预测情况，来评估这个模型。


预测新结果(可选)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

同样，知道如何预测性新的，不可知的数据也很重要。幸运的是，如果我们完成模型的训练之后，我们可以通过训练后的模型
来做这些事情。


总结
-------

在TensorFlow中，我们在程序进行训练并改变变量来预测变量之前，必须先建立数据，变量，占位符以及模型。 TensorFlow通过
计算图来完成这些。我们告诉它去最小化损失函数，而TensorFlow要通过改变变量来实现这一目标。TensorFlow知道如何改变变量，
这是因为它一直在关注模型的计算，然后自动计算每个变量的梯度。也正因为如此，我们也就知道改变它以及尝试不同数据的类型又
多么简单。

总的来说，算法在TensorFlow中会被设计成为循环的算法。我们把这个循环建成计算图，然后通过占位符来输入数据，计算计算图的
输出结果，用损失函数来比较输出结果，通过自动反向传播来改变模型中的变量，最后不断重复整个过程，直到达到设定的标准。

