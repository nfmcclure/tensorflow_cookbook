# Using TensorFlow with Keras
#----------------------------------
#
# This script will show you how to create model layers with Keras
#

import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# The following loads the MNIST dataset into
#
# mnist.[train/test].[images/labels]
#
# where images are a 1x784 flatt array and labels are an integer between 0 and 9.
#

mnist = input_data.read_data_sets("MNIST_data/")
x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_train = [[i] for i in y_train]
y_test = mnist.test.labels
y_test = [[i] for i in y_test]

# One-hot encode labels
one_hot = MultiLabelBinarizer()
y_train = one_hot.fit_transform(y_train)
y_test = one_hot.transform(y_test)

# Example 1: Fully connected neural network model
# We start with a 'sequential' model type (connecting layers together)
model = keras.Sequential()
# Adds a densely-connected layer with 32 units to the model, followed by an ReLU activation.
model.add(keras.layers.Dense(32, activation='relu'))
# Adds a densely-connected layer with 16 units to the model, followed by an ReLU activation.
model.add(keras.layers.Dense(16, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

# Train the model:
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Configure a model for mean-squared error regression.
# model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#              loss='mse',       # mean squared error
#              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
#model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#              loss=keras.losses.categorical_crossentropy,
#              metrics=[keras.metrics.categorical_accuracy])

# Fit the model:
model.fit(x_train,
          y_train,
          epochs=5,
          batch_size=64,
          validation_data=(x_test, y_test))


# ---------------------
# Simple CNN in Keras:
# ---------------------
# First we transform the input images from 1D arrays to 2D matrices. (28 x 28)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
num_classes = 10

# Categorize y targets
y_test = to_categorical(mnist.test.labels)
y_train = to_categorical(mnist.train.labels)

# Decrease test size for memory usage
x_test = x_test[:64]
y_test = y_test[:64]

# Start our sequential model
cnn_model = keras.Sequential()
cnn_model.add(keras.layers.Conv2D(25,
                                  kernel_size=(4, 4),
                                  strides=(1, 1),
                                  activation='relu',
                                  input_shape=input_shape))
cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2)))

cnn_model.add(keras.layers.Conv2D(50,
                                  kernel_size=(5, 5),
                                  strides=(1, 1),
                                  activation='relu'))

cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2)))

cnn_model.add(keras.layers.Flatten())

cnn_model.add(keras.layers.Dense(num_classes, activation='softmax'))

cnn_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

cnn_model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=3,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

print(history.acc)
