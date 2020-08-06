引言
----

Google在2015年11月完成了对TensorFlow的开源。自从那之后，`TensorFlow <https://github.com/tensorflow/tensorflow>`_ 
已经是Github上机器学习starred最多的仓库。

为什么选择TensorFlow ? TensorFlow的受欢迎程度归因于很多方面，但是主要是因为它的计算图概念，自动微分和TensorFlow的
Python API的架构。这些都使得程序员用TensorFlow来解决实际问题更加便捷。

Google的TensorFlow引擎有一个解决问题的独特方式。这种独特的方式使得解决机器学习问题非常有效。下面，我们会介绍TensorFlow
如何运行的基本步骤。

TensorFlow是如何运行的
-----------------------

在一开始的时候, TensorFlow中的计算可能看起来毫无必要的复杂. 但其实其中是有原因的: 也正因为TensorFlow处理计算的方式，发展
更为复杂的计算也就相对来说更为简单。这一节呢，会带领你领略一个TensorFlow算法通常工作的方式. 

现在呢，TensorFlow已经被所有的主流操作系统(Windows, Linux 和 Mac)所支持。通过这本书呢，我们只关心TensorFlow的Python库
这本书呢，会用到 `Python 3.x <https://www.python.org>`_ 和 `Tensorflow 0.12 + <https://www.tensorflow.org>`_ (我们这里会用
Python 3.7 和 TensorFlow 1.8 版本)。虽然说TensorFlow可以在CPU上运行，但是它在GPU(Graphic Processing Unit)运行得更快。
英伟达(Nvidia) Compute Capability 3.0+的显卡现在也支持TensorFlow。如果你想要在GPU上运行，你需要下载并安装 `NVidia Cuda Toolkit <https://developer.nvidia.com/cuda-downloads>`_。 有些章节可能还依赖安装Scipy, Numpy和Scikit-learn。你可以通过下载下面的requirements.txt, 然后运行下面的命令，
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
      
      >>> data = tf.nn.batch_norm_with_global_normalization(...)

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

定义模型结构
^^^^^^^^^^^^^^^^^^^^^^^^^^

在我们有了数据，并且将我们的变量和占位符进行初始化之后，我们就可以定义模型了。这个，我们可以通过建立一个计算图来完成。
我们告诉TensorFlow哪些操作需要在变量和占位符上完成，以实现我们的模型预测。关于计算图，我们会在第二章详细描述。
在这里我们，我们先看一下定义模型结构的例子：

.. code:: python
      
      >>> y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)


声明损失函数
^^^^^^^^^^^^^^^^^^^^^^^^^

在定义模型之后，我们就可以用TensorFlow算出结果了。这时候，我们需要定义一个损失函数。损失函数是非常重要的，
After defining the model, we must be able to evaluate the output. This is where we declare the loss function. 
The loss function is very important as it tells us how far off our predictions are from the actual values. 
The different types of loss functions are explored in greater detail in chapter two, section five.

.. code:: python
      
      >>> loss = tf.reduce_mean(tf.square(y_actual – y_pred))

Initialize and train the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have everything in place, we create an instance or our graph and feed in the data through the
placeholders and let Tensorflow change the variables to better predict our training data. Here is one way 
to initialize the computational graph.

.. code:: python
      
      >>> with tf.Session(graph=graph) as session:
               ...
      >>> session.run(...)
               ...

Note that we can also initiate our graph with

.. code:: python
      
      >>> session = tf.Session(graph=graph) session.run(…)

Evaluate the model(可选)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have built and trained the model, we should evaluate the model by looking at how well it does on 
new data through some specified criteria.

Predict new outcomes(可选)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also important to know how to make predictions on new, unseen, data. We can do this with all of 
our models, once we have them trained.

总结
-------

In Tensorflow, we have to setup the data, variables, placeholders, and model before we tell the program
to train and change the variables to improve the predictions. Tensorflow accomplishes this through the
computational graph. We tell it to minimize a loss function and Tensorflow does this by modifying the 
variables in the model. Tensorflow knows how to modify the variables because it keeps track of the 
computations in the model and automatically computes the gradients for every variable. Because of this,
we can see how easy it can be to make changes and try different data sources.

Overall, algorithms are designed to be cyclic in TensorFlow. We set up this cycle as a computational 
graph and (1) feed in data through the placeholders, (2) calculate the output of the computational graph, 
(3) compare the output to the desired output with a loss function, (4) modify the model variables 
according to the automatic back propagation, and finally (5) repeat the process until a stopping criteria is met.
