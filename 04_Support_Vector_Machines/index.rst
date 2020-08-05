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
       
       /04_Support_Vector_Machines/01_Introduction/index
   

We introduce the concept of SVMs and how we will go about implementing them in the TensorFlow framework.

------------

线性支持向量机
---------------
.. toctree::
       :maxdepth: 3
       
       /04_Support_Vector_Machines/02_Working_with_Linear_SVMs/index

We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris
data set.

.. image:: 

下载本章 :download:`Jupyter Notebook </04_Support_Vector_Machines/02_Working_with_Linear_SVMs/02_linear_svm.ipynb>`

-----

回归线性回归
--------------
.. toctree::
       :maxdepth: 3
       
       /04_Support_Vector_Machines/03_Reduction_to_Linear_Regression/index

The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly to perform SVM regression.

.. image:: 

下载本章 :download:`Jupyter Notebook </04_Support_Vector_Machines/03_Reduction_to_Linear_Regression/03_support_vector_regression.ipynb>`

-----------

TensorFlow中的核
----------
.. toctree::
       :maxdepth: 3
       
       /04_Support_Vector_Machines/04_Working_with_Kernels/index

In order to extend SVMs into non-linear data, we explain and show how to implement different kernels 
in TensorFlow.

下载本章 :download:`Jupyter Notebook </04_Support_Vector_Machines/04_Working_with_Kernels/04_svm_kernels.ipynb>`

-----------

非线性支持向量机
-------------
.. toctree::
       :maxdepth: 3
       
       /04_Support_Vector_Machines/05_Implementing_Nonlinear_SVMs/index

We use the Gaussian kernel (RBF) to separate non-linear classes.

下载本章 :download:`Jupyter Notebook </04_Support_Vector_Machines/05_Implementing_Nonlinear_SVMs/05_nonlinear_svm.ipynb>`

-------------

多类支持向量机
-----------

.. toctree::
       :maxdepth: 3
       
       /04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs/index

SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in 
TensorFlow.     

.. image::

下载本章 :download:`Jupyter Notebook </04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs/06_multiclass_svm.ipynb>`

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
