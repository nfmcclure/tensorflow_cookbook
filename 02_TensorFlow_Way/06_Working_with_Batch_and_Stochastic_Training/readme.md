# Working with Batch and Stochastic Training

## Summary

Here, we introduce the differences between batch and stochastic training and how to implement both in TensorFlow. Stochastic training is defined as training on one observation at once, and batch training is defined as training on a group of observations at once.

## Model
In this script, we use generated data.  We will generate input data that is distributed as Normal(mean=1, sd=0.1).  All the target data will be is the value 10.0 repeated.  The model will try to predict the multiplication factor to minimize the loss between the model output and the value 10.0.

## Notes

It is important to note that TensorFlow works well with many dimensional matrices, so we can easily implement batch training by adding the batch dimension to our inputs, as illustrated in this script.

## Viewing the Difference

Here we plot the loss function of stochastic and batch training on the same graph.  Notice how stochastic training is less smooth in the convergence of a solution.  This may sound like a bad thing, but it can help explore sample spaces and be less likely to be stuck in local minima.

![Stochastic and Batch](../images/06_Back_Propagation.png "Stochastic and Batch Loss")