# Using the Cholesky Decomposition Method

Here we implement solving 2D linear regression via the Cholesky decomposition in TensorFlow.

# Model

Given A * x = b, and a Cholesky decomposition such that A = L*L' then we can get solve for x via
 1. Solving L * y = t(A) * b for y
 2. Solving L' * x = y for x.

# Graph of linear fit

![Cholesky decomposition](../images/02_Cholesky_Decomposition.png "Cholesky decomposition")
