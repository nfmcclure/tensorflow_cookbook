.. note::

   After we have established the basic objects and methods in TensorFlow, we now want to 
   establish the components that make up TensorFlow algorithms.  We start by introducing 
   computational graphs, and then move to loss functions and back propagation.  We end with 
   creating a simple classifier and then show an example of evaluating regression and 
   classification algorithms.




计算图
----------------
.. toctree::
       :maxdepth: 3

We show how to create an operation on a computational graph and how to visualize it using Tensorboard.


下载本章 :download:`Jupyter Notebook </01_Introduction/01_How_TensorFlow_Works/01_How_TensorFlow_Works.ipynb>`

------------

分层嵌套操作
---------------
.. toctree::
       :maxdepth: 3


We show how to create multiple operations on a computational graph and how to visualize them using 
Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </01_Introduction/02_Creating_and_Using_Tensors/02_tensors.ipynb>`

-----

多层操作
--------------
.. toctree::
       :maxdepth: 3
       
Here we extend the usage of the computational graph to create multiple layers and show how they appear 
in Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </01_Introduction/03_Using_Variables_and_Placeholders/03_placeholders.ipynb>`

-----------

载入损失函数
----------
.. toctree::
       :maxdepth: 3
       

In order to train a model, we must be able to evaluate how well it is doing. This is given by loss functions.
We plot various loss functions and talk about the benefits and limitations of some.

下载本章 :download:`Jupyter Notebook </01_Introduction/04_Working_with_Matrices/04_matrices.ipynb>`

-----------

载入向后传递
-------------
.. toctree::
       :maxdepth: 3
       

Here we show how to use loss functions to iterate through data and back propagate errors for regression 
and classification.


下载本章 :download:`Jupyter Notebook </01_Introduction/05_Declaring_Operations/05_operations.ipynb>`

-------------

随机和批量训练
-----------

.. toctree::
       :maxdepth: 3
       

TensorFlow makes it easy to use both batch and stochastic training. We show how to implement both and talk 
about the benefits and limitations of each.

.. image::

下载本章 :download:`Jupyter Notebook </01_Introduction/06_Implementing_Activation_Functions/06_activation_functions.ipynb>`

-----------

结合训练
-------
.. toctree::
       :maxdepth: 3
       
We now combine everything together that we have learned and create a simple classifier.

下载本章 :download:`Jupyter Notebook </01_Introduction/07_Working_with_Data_Sources/07_data_gathering.ipynb>`

------------

模型评估
----------
.. toctree::
       :maxdepth: 3
  
Any model is only as good as it's evaluation.  Here we show two examples of (1) evaluating a regression 
algorithm and (2) a classification algorithm.




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





   
