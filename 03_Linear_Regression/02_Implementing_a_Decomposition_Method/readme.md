# Using the Cholesky Decomposition Method

Here we implement solving 2D linear regression via the Cholesky decomposition in Tensorflow.

# Model

Given A * x = b, and a Cholesky decomposition such that A = L*L' then we can get solve for x via
 1. Solving L * y = t(A) * b for y
 2. Solving L' * x = y for x.

# Graph of linear fit

![Cholesky decomposition](http://fromdata.org/wp-content/uploads/2016/07/B05480_03_02.png "Cholesky decomposition")
