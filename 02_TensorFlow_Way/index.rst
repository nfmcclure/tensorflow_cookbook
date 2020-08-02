After we have established the basic objects and methods in TensorFlow, we now want to
establish the components that make up TensorFlow algorithms.  We start by introducing 
computational graphs, and then move to loss functions and back propagation.  We end with 
creating a simple classifier and then show an example of evaluating regression and classification 
algorithms.

.. _my-reference-label:
计算图
===================

We show how to create an operation on a computational graph and how to visualize it using Tensorboard.

分层嵌套操作
==========

We show how to create multiple operations on a computational graph and how to visualize them using 
Tensorboard.

多层操作
=======
   
Here we extend the usage of the computational graph to create multiple layers and show how they appear 
in Tensorboard.
   
载入损失函数
============

In order to train a model, we must be able to evaluate how well it is doing. This is given by loss functions.
We plot various loss functions and talk about the benefits and limitations of some.

载入向后传递
============

Here we show how to use loss functions to iterate through data and back propagate errors for regression 
and classification.

随机和批量训练
=============

TensorFlow makes it easy to use both batch and stochastic training. We show how to implement both and talk 
about the benefits and limitations of each.


结合训练
============================

We now combine everything together that we have learned and create a simple classifier.


模型评估
==================
  
Any model is only as good as it's evaluation.  Here we show two examples of (1) evaluating a regression 
algorithm and (2) a classification algorithm.




   
