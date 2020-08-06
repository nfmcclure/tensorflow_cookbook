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

现在呢，TensorFlow已经被所有的主流操作系统(Windows, Linux 和 Mac)所支持。通过这本书呢，我们只关心TensorFlow的Python语言
Tensorflow is now supported on all three major OS systems (Windows, Linux, and Mac). Throughout this book we 
will only concern ourselves with the Python library wrapper of Tensorflow. This book will use 
`Python 3.X <https://www.python.org>`_ and `Tensorflow 0.12+ <https://www.tensorflow.org>`_. While Tensorflow can 
run on the CPU, it runs faster if it runs on the GPU, and it is supported on graphics cards with NVidia Compute
Capability 3.0+. To run on a GPU, you will also need to download and install the 
`NVidia Cuda Toolkit <https://developer.nvidia.com/cuda-downloads>`_. Some of the recipes will rely on a current 
installation of the Python packages Scipy, Numpy, and Scikit-Learn as well.

Please see the requirements.txt in the main directory of this repository and run a command similar to

.. code:: sh
      
      $ pip install -r requirements.txt 
      
to guarentee that all the necessary libraries are available.

General TensorFlow Algorithm Outlines
-------------------------------------
Here we will introduce the general flow of Tensorflow Algorithms. Most recipes will follow this outline.

Import or generate data
^^^^^^^^^^^^^^^^^^^^^^^^
All of our machine learning algorithms will depend on data. In this book we will either generate data or use 
an outside source of data. Sometimes it is better to rely on generated data because we will want to know the 
expected outcome. Other times we will access public data sets for the given recipe and details on accessing 
these are in section 8 of this chapter.

Transform and normalize data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The data is usually not in the correct dimension or type that our Tensorflow algorithms expect. We will have
to transform our data before we can use it. Most algorithms also expect normalized data and we will do this 
here as well. Tensorflow has built in functions that can normalize the data for you.

.. code:: python
      
      >>> data = tf.nn.batch_norm_with_global_normalization(...)

Set algorithm parameters
^^^^^^^^^^^^^^^^^^^^^^^
Our algorithms usually have a set of parameters that we hold constant throughout the procedure. For example, 
this can be the number of iterations, the learning rate, or other fixed parameters of our choosing. It is 
considered good form to initialize these together so the reader or user can easily find them.

.. code:: python
      
      >>> learning_rate = 0.01 
      >>> iterations = 1000

Initialize variables and placeholders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tensorflow depends on us telling it what it can and cannot modify. Tensorflow will modify the variables during 
optimization to minimize a loss function. To accomplish this, we feed in data through placeholders. We need to 
initialize both of these, variables and placeholders with size and type, so that Tensorflow knows what to expect.

.. code:: python
      
      >>> a_var = tf.constant(42) 
      >>> x_input = tf.placeholder(tf.float32, [None, input_size]) 
      >>> y_input = tf.placeholder(tf.float32, [None, num_classes])

Define the model structure
^^^^^^^^^^^^^^^^^^^^^^^^^^
After we have the data, and initialized our variables and placeholders, we have to define the model. This is 
done by building a computational graph. We tell Tensorflow what operations must be done on the variables and
placeholders to arrive at our model predictions. We talk more in depth about computational graphs in chapter two, 
section one of this book.

.. code:: python
      
      >>> y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)


Declare the loss functions
^^^^^^^^^^^^^^^^^^^^^^^^^
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

(Optional) Evaluate the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have built and trained the model, we should evaluate the model by looking at how well it does on 
new data through some specified criteria.

(Optional) Predict new outcomes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also important to know how to make predictions on new, unseen, data. We can do this with all of 
our models, once we have them trained.

Summary
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
