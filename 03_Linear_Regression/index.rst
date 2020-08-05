.. note::
  
  Here we show how to implement various linear regression techniques in TensorFlow.  
  The first two sections show how to do standard matrix linear regression solving in 
  TensorFlow.  The remaining six sections depict how to implement various types of 
  regression using computational graphs in TensorFlow.


矩阵转置
----------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/01_Operations_as_a_Computational_Graph/index
       
       
How to solve a 2D regression with a matrix inverse in TensorFlow.

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/01_Operations_as_a_Computational_Graph/01_operations_on_a_graph.ipynb>`

------------

矩阵分解法
---------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/02_Layering_Nested_Operations/index


Solving a 2D linear regression with Cholesky decomposition.


.. image:: 

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/02_Layering_Nested_Operations/02_layering_nested_operations.ipynb>`

-----

TensorFLow的线性回归
--------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/03_Working_with_Multiple_Layers/index

Linear regression iterating through a computational graph with L2 Loss.
Here we extend the usage of the computational graph to create multiple layers and show how they appear 
in Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/03_Working_with_Multiple_Layers/03_multiple_layers.ipynb>`

-----------

线性回归的损失函数
----------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/04_Implementing_Loss_Functions/index

L2 vs L1 loss in linear regression.  We talk about the benefits and limitations of
both.


下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/04_Implementing_Loss_Functions/04_loss_functions.ipynb>`

-----------

Deming回归(全回归)
-------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/05_Implementing_Back_Propagation/index

Deming (total) regression implemented in TensorFlow by changing the loss function.


下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/05_Implementing_Back_Propagation/05_back_propagation.ipynb>`

-------------

套索(Lasso)回归和岭(Ridge)回归
-----------

.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/06_Working_with_Batch_and_Stochastic_Training/index
       
Lasso and Ridge regression are ways of regularizing the coefficients. We implement 
both of these in TensorFlow via changing the loss functions.

.. image::

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/06_Working_with_Batch_and_Stochastic_Training/06_batch_stochastic_training.ipynb>`

-----------

弹性网(Elastic Net)回归
-------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/07_Combining_Everything_Together/index

Elastic net is a regularization technique that combines the L2 and L1 loss for coefficients. 
We show how to implement this in TensorFlow.

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/07_Combining_Everything_Together/07_combining_everything_together.ipynb>`

------------

逻辑(Logistic)回归
----------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/08_Evaluating_Models/index

We implement logistic regression by the use of an activation function in our computational graph.


下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/08_Evaluating_Models/08_evaluating_models.ipynb>`


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





