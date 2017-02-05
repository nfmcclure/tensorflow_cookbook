# Implementing Deming Regression

Deming regression, also known as total regression, is regular regression that minimizes the shortest distance to the line. Contrast this to regular regression, in which we aim to minimize the vertical distance between the model output and the y-target values.

![Deming Regression](../images/05_demming_vs_linear_reg.png "Deming Regression")

# Model

The model will be the same as regular linear regression:

y = A * x + b

Instead of measuring the vertical L2 distance, we will measure the shortest distance between the line and the predicted point in the loss function.

loss = |y\_target - (A * x\_input + b)| / sqrt(A^2 + 1)

# Graph of Linear Fit

![Deming Output](../images/05_demming_reg.png "Deming Output")
