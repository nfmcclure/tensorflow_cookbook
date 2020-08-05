.. note::

  This chapter shows how to implement various SVM methods with TensorFlow.  We first
  create a linear SVM and also show how it can be used for regression.  We then introduce
  kernels (RBF Gaussian kernel) and show how to use it to split up non-linear data. We
  finish with a multi-dimensional implementation of non-linear SVMs to work with multiple
  classes.  

引言
----------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/01_Using_the_Matrix_Inverse_Method/index
   

We introduce the concept of SVMs and how we will go about implementing them in the TensorFlow 
framework.

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/01_Using_the_Matrix_Inverse_Method/01_lin_reg_inverse.ipynb>`

------------

线性支持向量机
---------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/02_Implementing_a_Decomposition_Method/index

We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris
data set.

.. image:: 

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/02_Implementing_a_Decomposition_Method/02_lin_reg_decomposition.ipynb>`

-----

回归线性回归
--------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/03_TensorFlow_Way_of_Linear_Regression/index

The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly
to perform SVM regression.

.. image:: 

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/03_TensorFlow_Way_of_Linear_Regression/03_lin_reg_tensorflow_way.ipynb>`

-----------

TensorFlow中的核
----------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions/index

In order to extend SVMs into non-linear data, we explain and show how to implement different kernels 
in TensorFlow.

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions/04_lin_reg_l1_vs_l2.ipynb>`

-----------

非线性支持向量机
-------------
.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/05_Implementing_Deming_Regression/index

We use the Gaussian kernel (RBF) to separate non-linear classes.

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/05_Implementing_Deming_Regression/05_deming_regression.ipynb>`

-------------

多类支持向量机
-----------

.. toctree::
       :maxdepth: 3
       
       /03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/index

SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in 
TensorFlow.     

.. image::

下载本章 :download:`Jupyter Notebook </03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.ipynb>`

-----------

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
