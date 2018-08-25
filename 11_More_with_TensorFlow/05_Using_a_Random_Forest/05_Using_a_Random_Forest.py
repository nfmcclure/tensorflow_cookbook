"""
Using a Random Forest
---------------------

This script will illustrate how to use TensorFlow's Boosted Random Forest algorithm.


For illustrative purposes we will show how to do this with the boston housing data.

Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                 by town
    13. LSTAT    % lower status of the population
    14. y_target Median value of owner-occupied homes in $1000's.
"""

import os
import numpy as np
import tensorflow as tf
from keras.datasets import boston_housing
from tensorflow.python.framework import ops
ops.reset_default_graph()

# For using the boosted trees classifier (binary classification) in TF:
# Note: target labels have to be 0 and 1.
boosted_classifier = tf.estimator.BoostedTreesClassifier

# For using a boosted trees regression classifier (binary classification) in TF:
regression_classifier = tf.estimator.BoostedTreesRegressor

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Set model parameters
# Batch size
batch_size = 32
# Number of training steps
train_steps = 500
# Number of trees in our 'forest'
n_trees = 100
# Maximum depth of any tree in forest
max_depth = 6

# Data ETL
binary_split_cols = ['CHAS', 'RAD']
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X_dtrain = {col: x_train[:, ix] for ix, col in enumerate(col_names)}
X_dtest = {col: x_test[:, ix] for ix, col in enumerate(col_names)}

# Create feature columns!
feature_cols = []
for ix, column in enumerate(x_train.T):
    col_name = col_names[ix]

    # Create binary split feature
    if col_name in binary_split_cols:
        # To create 2 buckets, need 1 boundary - the mean
        bucket_boundaries = [column.mean()]
        numeric_feature = tf.feature_column.numeric_column(col_name)
        final_feature = tf.feature_column.bucketized_column(source_column=numeric_feature, boundaries=bucket_boundaries)
    # Create bucketed feature
    else:
        # To create 5 buckets, need 4 boundaries
        bucket_boundaries = list(np.linspace(column.min() * 1.1, column.max() * 0.9, 4))
        numeric_feature = tf.feature_column.numeric_column(col_name)
        final_feature = tf.feature_column.bucketized_column(source_column=numeric_feature, boundaries=bucket_boundaries)

    # Add feature to feature_col list
    feature_cols.append(final_feature)


# Create an input function
input_fun = tf.estimator.inputs.numpy_input_fn(X_dtrain, y=y_train, batch_size=batch_size, num_epochs=10, shuffle=True)

# Training
model = regression_classifier(feature_columns=feature_cols,
                              n_trees=n_trees,
                              max_depth=max_depth,
                              learning_rate=0.25,
                              n_batches_per_layer=batch_size)
model.train(input_fn=input_fun, steps=train_steps)

# Evaluation on test set
# Do not shuffle when predicting
p_input_fun = tf.estimator.inputs.numpy_input_fn(X_dtest, y=y_test, batch_size=batch_size, num_epochs=1, shuffle=False)
# Get predictions
predictions = list(model.predict(input_fn=p_input_fun))
final_preds = [pred['predictions'][0] for pred in predictions]

# Get accuracy (mean absolute error, MAE)
mae = np.mean([np.abs((actual - predicted) / predicted) for actual, predicted in zip(y_test, final_preds)])
print('Mean Abs Err on test set: {}'.format(acc))
