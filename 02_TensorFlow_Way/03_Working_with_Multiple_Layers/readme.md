# Working with Multiple Layers

## Summary

In this script, we will perform a 1D spatial moving average across a vector.  Then we will create a "custom" operation by multiplying the output by a specific matrix:

## The Spatial Moving Window Layer
We will create a layer that takes a spatial moving window average. Our window will be 2x2 with a stride of 2 for height and width. The filter value will be 0.25 because we want the average of the 2x2 window
```
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')
```

## Custom Layer

We create a custom layer which will be sigmoid(Ax+b) where x is a 2x2 matrix and A and b are 2x2 matrices.

```
output = sigmoid( input * A + b )
```

## Computational Graph Output

Viewing the computational graph in Tensorboard:

![Multiple Layers](../images/03_Multiple_Layers.png "Multiple Layers on a Graph")