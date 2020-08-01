Here we show how to implement various linear regression techniques in TensorFlow.  
The first two sections show how to do standard matrix linear regression solving in 
TensorFlow.  The remaining six sections depict how to implement various types of 
regression using computational graphs in TensorFlow.

矩阵转置
=======

How to solve a 2D regression with a matrix inverse in TensorFlow.

矩阵分解法
=========

Solving a 2D linear regression with Cholesky decomposition.

TensorFLow的线性回归
===================

Linear regression iterating through a computational graph with L2 Loss.

线性回归的损失函数
=================

L2 vs L1 loss in linear regression.  We talk about the benefits and limitations of
both.

Deming回归(全回归)
===================

Deming (total) regression implemented in TensorFlow by changing the loss function.

套索(Lasso)回归和岭(Ridge)回归
===============================

Lasso and Ridge regression are ways of regularizing the coefficients. We implement 
both of these in TensorFlow via changing the loss functions.

弹性网(Elastic Net)回归
=========================

Elastic net is a regularization technique that combines the L2 and L1 loss for coefficients. 
We show how to implement this in TensorFlow.

逻辑(Logistic)回归
==================

We implement logistic regression by the use of an activation function in our computational graph.

