## [Ch 4: Support Vector Machines](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines)

This chapter shows how to implement various SVM methods with Tensorflow.  We first create a linear SVM and also show how it can be used for regression.  We then introduce kernels (RBF Gaussian kernel) and show how to use it to split up non-linear data. We finish with a multi-dimensional implementation of non-linear SVMs to work with multiple classes.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/01_Introduction)
  * We introduce the concept of SVMs and how we will go about implementing them in the Tensorflow framework.
 2. [Working with Linear SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/02_Working_with_Linear_SVMs)
  * We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris data set.
 3. [Reduction to Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/03_Reduction_to_Linear_Regression)
  * The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly to perform SVM regression.
 4. [Working with Kernels in Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/04_Working_with_Kernels)
  * In order to extend SVMs into non-linear data, we explain and show how to implement different kernels in Tensorflow.
 5. [Implmenting Non-Linear SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/05_Implementing_Nonlinear_SVMs)
  * We use the Gaussian kernel (RBF) to separate non-linear classes.
 6. [Implementing Multi-class SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs)
  * SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in Tensorflow.
