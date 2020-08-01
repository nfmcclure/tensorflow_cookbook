This chapter shows how to implement various SVM methods with TensorFlow.  We first
create a linear SVM and also show how it can be used for regression.  We then introduce
kernels (RBF Gaussian kernel) and show how to use it to split up non-linear data. We
finish with a multi-dimensional implementation of non-linear SVMs to work with multiple
classes.  

引言
=====

We introduce the concept of SVMs and how we will go about implementing them in the TensorFlow 
framework.

线性支持向量机
==============

We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris
data set.

回归线性回归
=============

The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly
to perform SVM regression.

TensorFlow中的核
=================

In order to extend SVMs into non-linear data, we explain and show how to implement different kernels 
in TensorFlow.

非线性支持向量机
==============

We use the Gaussian kernel (RBF) to separate non-linear classes.

多类支持向量机
=============

SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in 
TensorFlow.
