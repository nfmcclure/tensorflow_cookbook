.. image:: https://readthedocs.org/projects/tensorflow-ml/badge/?version=latest  
.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat

.. TensorFlow-ML documentation master file, created by
   Wei MEI on 1st August 20:30:46 2020.

TensorFlow 机器学习 Cookbook
============================

TensorFlow在2015年的时候已经成为开源项目, 自从那之后它已经成为Github中starred最多的机器学习库.
TensorFlow的受欢迎度主要归功于它能帮助程序员创造计算图(computational graphs), 自动微分 (automatic
differentation) 和 可定制性 (customizability). 由于这些特性，TensorFlow是一个强有力的灵活性高的工具, 
用于解决很多机器学习的问题. 

本教程阐述很多机器学习算法, 以及如何把它们应用到实际情况中, 以及如何诠释所得到的结果.

.. important::
   
   - 第一章: 从TensorFlow开始, 介绍主要tensorflow的对象与概念. 我们介绍张量, 变量和占位符. 我们也会展示如何在tensorflow中使用矩阵和其他的数学操作. 在本章的末尾，我们会展示如何获取数据资源.
   
   - 第二章: TensorFlow算法, 阐述如何用多种方式将第一章中所有的算法成分关联成一个计算图并创造出一个简单的分类器. 在阐述的过程中, 我们会介绍计算图(computational graphs), 损失函数(loss functions), 反向传播(back propagation), 以及训练数据.
   
   - 第三章: 线性回归 (Linear Regression), 本章着重强调如何使用tensorflow来探索不同的线性回归技巧, 比如Deming, lasso, ridge, elastic net 和 logistic regression. 我们会在计算图中展示如何应用它们.
   
   - 

.. toctree::
   :maxdepth: 2
   :caption: 从TensorFlow开始 (Getting Started)
   
   01_Introduction/index

.. toctree::
   :maxdepth: 2
   :caption: TensorFlow算法 (TensorFlow Way)
   
   02_TensorFlow_Way/index
   
.. toctree::
   :maxdepth: 2
   :caption: 线性回归 (Linear Regression)
   
   03_Linear_Regression/index
   
.. toctree::
   :maxdepth: 2
   :caption: 支持向量机(Support Vector Machines)
   
   04_Support_Vector_Machines/index
   
.. toctree::
   :maxdepth: 2
   :caption: 最近邻法 (Nearest Neighbor Methods)
   
   05_Nearest_Neighbor_Methods/index
   
.. toctree::
   :maxdepth: 2
   :caption: 神经元网络 (Neural Networks)
   
   06_Neural_Networks/index
   
.. toctree::
   :maxdepth: 2
   :caption: Natural Language Processing
   
   07_Natural_Language_Processing/index
   
.. toctree::
   :maxdepth: 2
   :caption: Convolutional Neural Networks
   
   08_Convolutional_Neural_Networks/index

.. toctree::
   :maxdepth: 2
   :caption: Recurrent Neural Networks
   
   09_Recurrent_Neural_Networks/index

.. toctree::
   :maxdepth: 2
   :caption: Taking TensorFlow to Production
   
   10_Taking_TensorFlow_to_Production/index

.. toctree::
   :maxdepth: 2
   :caption: More with TensorFlow
   
   11_More_with_TensorFlow/index
