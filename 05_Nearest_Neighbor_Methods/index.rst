Nearest Neighbor methods are a very popular ML algorithm.  We show how to implement k-Nearest 
Neighbors, weighted k-Nearest Neighbors, and k-Nearest Neighbors with mixed distance functions. 
In this chapter we also show how to use the Levenshtein distance (edit distance) in TensorFlow, 
and use it to calculate the distance between strings. We end this chapter with showing how to 
use k-Nearest Neighbors for categorical prediction with the MNIST handwritten digit recognition.

引言
=====

We introduce the concepts and methods needed for performing k-Nearest Neighbors in TensorFlow.

最近邻法的使用
=============

We create a nearest neighbor algorithm that tries to predict housing worth (regression).

文本距离函数
============

In order to use a distance function on text, we show how to use edit distances in TensorFlow.

计算混合距离函数
===============

Here we implement scaling of the distance function by the standard deviation of the input 
feature for k-Nearest Neighbors.

地址匹配
========

We use a mixed distance function to match addresses. We use numerical distance for zip codes,
and string edit distance for street names. The street names are allowed to have typos.

图像处理的近邻法
==============
   
The MNIST digit image collection is a great data set for illustration of how to perform 
k-Nearest Neighbors for an image classification task.

