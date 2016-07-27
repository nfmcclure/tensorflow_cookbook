## [Ch 6: Neural Networks](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks)

Neural Networks are very important in machine learning and growing in popularity due to the major breakthroughs in prior unsolved problems.  We must start with introducing 'shallow' neural networks, which are very powerful and can help us improve our prior ML algorithm results.  We start by introducing the very basic NN unit, the operational gate.  We gradually add more and more to the neural network and end with training a model to play tic-tac-toe.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/01_Introduction)
  * We introduce the concept of neural networks and how Tensorflow is built to easily handle these algorithms.
 2. [Implementing Operational Gates](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/02_Implementing_an_Operational_Gate)
  * We implement an operational gate with one operation. Then we show how to extend this to multiple nested operations.
 3. [Working with Gates and Activation Functions](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/03_Working_with_Activation_Functions)
  * Now we have to introduce activation functions on the gates.  We show how different activation functions operate.
 4. [Implmenting a One Layer Neural Network](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/04_Single_Hidden_Layer_Network)
  * We have all the pieces to start implementing our first neural network.  We do so here with regression on the Iris data set.
 5. [Implementing Different Layers](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/05_Implementing_Different_Layers)
  * This section introduces the convolution layer and the max-pool layer.  We show how to chain these together in a 1D and 2D example with fully connected layers as well.
 6. [Using Multi-layer Neural Networks](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/06_Using_Multiple_Layers)
  * Here we show how to functionalize different layers and variables for a cleaner multi-layer neural network.
 7. [Improving Predictions of Linear Models](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/07_Improving_Linear_Regression)
  * We show how we can improve the convergence of our prior logistic regression with a set of hidden layers.
 8. [Learning to Play Tic-Tac-Toe](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/08_Learning_Tic_Tac_Toe)
  * Given a set of tic-tac-toe boards and corresponding optimal moves, we train a neural network classification model to play.  At the end of the script, we can attempt to play against the trained model.
