.. note::

   Neural Networks are very important in machine learning and growing in popularity due to the major 
   breakthroughs in prior unsolved problems.  We must start with introducing 'shallow' neural networks, 
   which are very powerful and can help us improve our prior ML algorithm results.  We start by introducing 
   the very basic NN unit, the operational gate.  We gradually add more and more to the neural network 
   and end with training a model to play tic-tac-toe.
   
   
引言
----------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/01_Introduction/index
 
We introduce the concept of neural networks and how TensorFlow is built to easily handle these algorithms.

------------

载入操作门
---------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/02_Implementing_an_Operational_Gate/index

We implement an operational gate with one operation. Then we show how to extend this to multiple nested 
operations.


.. image:: 

下载本章 :download:`Jupyter Notebook </06_Neural_Networks/02_Implementing_an_Operational_Gate/02_gates.ipynb>`

-----

门运算和激活函数
--------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/03_Working_with_Activation_Functions/index

Now we have to introduce activation functions on the gates.  We show how different activation functions 
operate.

.. image:: 

下载本章 :download:`Jupyter Notebook </06_Neural_Networks/03_Working_with_Activation_Functions/03_activation_functions.ipynb>`

-----------

载入一层神经网络
----------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/index

We have all the pieces to start implementing our first neural network.  We do so here with regression on
the Iris data set.

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/04_mixed_distance_functions_knn.ipynb>`

-----------

载入多层神经网络
-------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/index

This section introduces the convolution layer and the max-pool layer.  We show how to chain these together
in a 1D and 2D example with fully connected layers as well.


下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/05_address_matching.ipynb>`

-------------

使用多层神经网络
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index

Here we show how to functionalize different layers and variables for a cleaner multi-layer neural network.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`

-----------

线性模型预测改善
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index

We show how we can improve the convergence of our prior logistic regression with a set of hidden layers.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`


神经网络学习井字棋
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index

Given a set of tic-tac-toe boards and corresponding optimal moves, we train a neural network classification
model to play.  At the end of the script, we can attempt to play against the trained model.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`



本章学习模块
-----------

.. Submodules
.. ----------

*tensorflow\.zeros* 
^^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.zeros
    :members:
    :undoc-members:
    :show-inheritance:

------

*tensorflow\.ones*
^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.ones
    :members:
    :undoc-members:
    :show-inheritance:

-------------
