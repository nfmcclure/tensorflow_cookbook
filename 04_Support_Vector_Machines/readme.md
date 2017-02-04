## [Ch 4: Support Vector Machines](#ch-4-support-vector-machines)

This chapter shows how to implement various SVM methods with TensorFlow.  We first create a linear SVM and also show how it can be used for regression.  We then introduce kernels (RBF Gaussian kernel) and show how to use it to split up non-linear data. We finish with a multi-dimensional implementation of non-linear SVMs to work with multiple classes.

 1. [Introduction](01_Introduction)
  * We introduce the concept of SVMs and how we will go about implementing them in the TensorFlow framework.
 2. [Working with Linear SVMs](02_Working_with_Linear_SVMs)
  * We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris data set.
 3. [Reduction to Linear Regression](03_Reduction_to_Linear_Regression)
  * The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly to perform SVM regression.
 4. [Working with Kernels in TensorFlow](04_Working_with_Kernels)
  * In order to extend SVMs into non-linear data, we explain and show how to implement different kernels in TensorFlow.
 5. [Implmenting Non-Linear SVMs](05_Implementing_Nonlinear_SVMs)
  * We use the Gaussian kernel (RBF) to separate non-linear classes.
 6. [Implementing Multi-class SVMs](06_Implementing_Multiclass_SVMs)
  * SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in TensorFlow.
