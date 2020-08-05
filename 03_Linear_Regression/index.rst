.. note::
  
  Here we show how to implement various linear regression techniques in TensorFlow.  
  The first two sections show how to do standard matrix linear regression solving in 
  TensorFlow.  The remaining six sections depict how to implement various types of 
  regression using computational graphs in TensorFlow.


矩阵转置
----------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/01_Using_the_Matrix_Inverse_Method/index
       
       
How to solve a 2D regression with a matrix inverse in TensorFlow.

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/01_Using_the_Matrix_Inverse_Method/01_lin_reg_inverse.ipynb>`

------------

矩阵分解法
---------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/02_Implementing_a_Decomposition_Method/index


Solving a 2D linear regression with Cholesky decomposition.


.. image:: 

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/02_Implementing_a_Decomposition_Method/02_lin_reg_decomposition.ipynb>`

-----

TensorFLow的线性回归
--------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/03_TensorFlow_Way_of_Linear_Regression/index

Linear regression iterating through a computational graph with L2 Loss.
Here we extend the usage of the computational graph to create multiple layers and show how they appear 
in Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/03_TensorFlow_Way_of_Linear_Regression/03_lin_reg_tensorflow_way.ipynb>`

-----------

线性回归的损失函数
----------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions/index

L2 vs L1 loss in linear regression.  We talk about the benefits and limitations of
both.


下载本章 :download:`Jupyter Notebook </03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions/04_lin_reg_l1_vs_l2.ipynb>`

-----------

Deming回归(全回归)
-------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/05_Implementing_Deming_Regression/index

Deming (total) regression implemented in TensorFlow by changing the loss function.


下载本章 :download:`Jupyter Notebook </03_Linear_Regression/05_Implementing_Deming_Regression/05_deming_regression.ipynb>`

-------------

套索(Lasso)回归和岭(Ridge)回归
-----------

.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/index
       
Lasso and Ridge regression are ways of regularizing the coefficients. We implement 
both of these in TensorFlow via changing the loss functions.

.. image::

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb>`

-----------

弹性网(Elastic Net)回归
-------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/07_Implementing_Elasticnet_Regression/index

Elastic net is a regularization technique that combines the L2 and L1 loss for coefficients. 
We show how to implement this in TensorFlow.

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/07_Implementing_Elasticnet_Regression/07_elasticnet_regression.ipynb>`

------------

逻辑(Logistic)回归
----------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/08_Implementing_Logistic_Regression/index

We implement logistic regression by the use of an activation function in our computational graph.


下载本章 :download:`Jupyter Notebook </03_Linear_Regression/08_Implementing_Logistic_Regression/08_logistic_regression.ipynb>`


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





