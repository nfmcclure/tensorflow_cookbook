# Working with Linear SVMs

We introduce a linear SVM on a binary set, which will be a subset of the Iris data.  We know for I. setosa, that petal width and sepal length are completely seperable. We will create a linear SVM to predict I. setosa based on two features: petal width and sepal length.

It is worth noting that due to the small data set and the randomness of separating into train/test sets, that it may appear that a few points can end up on the wrong side of the line.  This is because they are in the test set, and this will result in a lower test accuracy.

# Model

We will aim to maximize the margin width, 2/||A||, or minimize ||A||.  We allow for a soft margin by having an error term in the loss function which is the max(0, 1-pred*actual).

![Linear Separator](http://fromdata.org/wp-content/uploads/2016/07/B05480_04_01.png "Linear Separator")

# Graph of Linear SVM

Here is a plot of the linear SVM separator of I. setosa based on petal width and sepal length.

![Linear SVM Output](http://fromdata.org/wp-content/uploads/2016/07/B05480_04_02.png "Linear SVM Output")

The accuracy is below, plotted over each iteration.

![Linear SVM Accuracy](http://fromdata.org/wp-content/uploads/2016/07/B05480_04_03.png "Linear SVM Accuracy")
