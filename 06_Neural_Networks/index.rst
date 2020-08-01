神经元网络 (Neural Networks)
===========================

Neural Networks are very important in machine learning and growing in popularity due to the major 
breakthroughs in prior unsolved problems.  We must start with introducing 'shallow' neural networks, 
which are very powerful and can help us improve our prior ML algorithm results.  We start by introducing 
the very basic NN unit, the operational gate.  We gradually add more and more to the neural network 
and end with training a model to play tic-tac-toe.
   
引言
----

We introduce the concept of neural networks and how TensorFlow is built to easily handle these algorithms.

Implementing Operational Gates
------------------------------

We implement an operational gate with one operation. Then we show how to extend this to multiple nested 
operations.

Working with Gates and Activation Functions
--------------------------------------------

Now we have to introduce activation functions on the gates.  We show how different activation functions 
operate.

Implementing a One Layer Neural Network
----------------------------------------

We have all the pieces to start implementing our first neural network.  We do so here with regression on
the Iris data set.

Implementing Different Layers
-----------------------------

This section introduces the convolution layer and the max-pool layer.  We show how to chain these together
in a 1D and 2D example with fully connected layers as well.

Using Multi-layer Neural Networks
----------------------------------

Here we show how to functionalize different layers and variables for a cleaner multi-layer neural network.

Improving Predictions of Linear Models
--------------------------------------

We show how we can improve the convergence of our prior logistic regression with a set of hidden layers.

Learning to Play Tic-Tac-Toe
------------------------------

Given a set of tic-tac-toe boards and corresponding optimal moves, we train a neural network classification
model to play.  At the end of the script, we can attempt to play against the trained model.
