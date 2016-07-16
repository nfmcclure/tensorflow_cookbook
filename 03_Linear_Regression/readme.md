## [Ch 3: Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression)

Here we show how to implement various linear regression techniques in Tensorflow.  The first two sections show how to do standard matrix linear regression solving in Tensorflow.  The remaining six sections depict how to implement various types of regression using computational graphs in Tensorflow.

 1. [Using the Matrix Inverse Method](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/01_Using_the_Matrix_Inverse_Method)
  * How to solve a 2D regression with a matrix inverse in Tensorflow.
 2. [Implementing a Decomposition Method](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/02_Implementing_a_Decomposition_Method)
  * Solving a 2D linear regression with Cholesky decomposition.
 3. [Learning the Tensorflow Way of Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/03_Tensorflow_Way_of_Linear_Regression)
  * Linear regression iterating through a computational graph with L2 Loss.
 4. [Understanding Loss Functions in Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions)
  * L2 vs L1 loss in linear regression.  We talk about the benefits and limitations of both.
 5. [Implementing Deming Regression (Total Regression)](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/05_Implementing_Deming_Regression)
  * Deming (total) regression implmented in Tensorflow by changing the loss function.
 6. [Implementing Lasso and Ridge Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression)
  * Lasso and Ridge regression are ways of regularizing the coefficients. We implement both of these in Tensorflow via changing the loss functions.
 7. [Implementing Elastic Net Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/07_Implementing_Elasticnet_Regression)
  * Elastic net is a regularization technique that combines the L2 and L1 loss for coefficients.  We show how to implement this in Tensorflow.
 8. [Implementing Logistic Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/08_Implementing_Logistic_Regression)
  * We implment logistic regression by the use of an activation function in our computational graph.
