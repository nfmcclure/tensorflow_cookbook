<img src="http://fromdata.org/wp-content/uploads/2016/07/B05480_MockupCover_Normal_New.jpg" data-canonical-src="http://fromdata.org/wp-content/uploads/2016/07/B05480_MockupCover_Normal_New.jpg" width="200" height="250" />

# [Tensorflow Machine Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook)
## [A Packt Publishing Book due out Jan. 2017.](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook)

### By Nick McClure
===================

[<a href="https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook">Tensorflow Machine Learning Cookbook</a>](#tensorflow-machine-learning-cookbook)

Table of Contents
=================

  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction">Ch 1: Getting Started with Tensorflow</a>](#ch-1-getting-started-with-tensorflow)

<kbd><a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/01_How_Tensorflow_Works"><img src="https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/images/01_outline.png" align="center" height="24" width="48" ></a></kbd>
<kbd><a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/02_Creating_and_Using_Tensors"><img src="https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/images/02_variable.png" align="center" height="24" width="48" ></a></kbd>
<kbd><a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/03_Using_Variables_and_Placeholders"><img src="https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/images/03_placeholder.png" align="center" height="24" width="48" ></a></kbd>
<kbd><a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/06_Implementing_Activation_Functions"><img src="https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/images/06_activation_funs1.png" align="center" height="24" width="48" ></a></kbd>
<kbd><a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/06_Implementing_Activation_Functions"><img src="https://github.com/nfmcclure/tensorflow_cookbook/blob/master/01_Introduction/images/06_activation_funs2.png" align="center" height="24" width="48" ></a></kbd>

  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way">Ch 2: The Tensorflow Way</a>](#ch-2-the-tensorflow-way)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression">Ch 3: Linear Regression</a>](#ch-3-linear-regression)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines">Ch 4: Support Vector Machines</a>](#ch-4-support-vector-machines)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods">Ch 5: Nearest Neighbor Methods</a>](#ch-5-nearest-neighbor-methods)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks">Ch 6: Neural Networks</a>](#ch-6-neural-networks)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing">Ch 7: Natural Language Processing</a>](#ch-7-natural-language-processing)
  * [<a href="https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks">Ch 8: Convolutional Neural Networks</a>](#ch-8-convolutional-neural-networks)
  * [Ch 9: Recurrent Neural Networks](#ch-9-recurrent-neural-networks)
  * [Ch 10: Taking Tensorflow to Production](#ch-10-taking-tensorflow-to-production)
  * [Ch 11: More with Tensorflow](#ch-11-more-with-tensorflow)


---

## [Ch 1: Getting Started with Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction)


This chapter intends to introduce the main objects and concepts in Tensorflow.  We also introduce how to access the data for the rest of the book and provide additional resources for learning about Tensorflow.

 1. [General Outline of TF Algorithms](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/01_How_Tensorflow_Works)
  * Here we introduce Tensorflow and the general outline of how most Tensorflow algorithms work.
 2. [Creating and Using Tensors](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/02_Creating_and_Using_Tensors)
  * How to create and initialize tensors in Tensorflow.  We also depict how these operations appear in Tensorboard.
 3. [Using Variables and Placeholders](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/03_Using_Variables_and_Placeholders)
  * How to create and use variables and placeholders in Tensorflow.  We also depict how these operations appear in Tensorboard.
 4. [Working with Matrices](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/04_Working_with_Matrices)
  * Understanding how Tensorflow can work with matrices is crucial to understanding how the algorithms work.
 5. [Declaring Operations](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/05_Declaring_Operations)
  * How to use various mathematical operations in Tensorflow.
 6. [Implementing Activation Functions](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/06_Implementing_Activation_Functions)
  * Activation functions are unique functions that Tensorflow has built in for your use in algorithms.
 7. [Working with Data Sources](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/07_Working_with_Data_Sources)
  * Here we show how to access all the various required data sources in the book.  There are also links describing the data sources and where they come from.
 8. [Additional Resources](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/01_Introduction/08_Additional_Resources)
  * Mostly official resources and papers.  The papers are Tensorflow papers or Deep Learning resources.

## [Ch 2: The Tensorflow Way](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way)

After we have established the basic objects and methods in Tensorflow, we now want to establish the components that make up Tensorflow algorithms.  We start by introducing computational graphs, and then move to loss functions and back propagation.  We end with creating a simple classifier and then show an example of evaluating regression and classification algorithms.

 1. [One Operation as a Computational Graph](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/01_Operations_as_a_Computational_Graph)
  * We show how to create an operation on a computational graph and how to visualize it using Tensorboard.
 2. [Layering Nested Operations](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/02_Layering_Nested_Operations)
  * We show how to create multiple operations on a computational graph and how to visualize them using Tensorboard.
 3. [Working with Multiple Layers](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/03_Working_with_Multiple_Layers)
  * Here we extend the usage of the computational graph to create multiple layers and show how they appear in Tensorboard.
 4. [Implmenting Loss Functions](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/04_Implementing_Loss_Functions)
  * In order to train a model, we must be able to evaluate how well it is doing. This is given by loss functions. We plot various loss functions and talk about the benefits and limitations of some.
 5. [Implmenting Back Propagation](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/05_Implementing_Back_Propagation)
  * Here we show how to use loss functions to iterate through data and back propagate errors for regression and classification.
 6. [Working with Stochastic and Batch Training](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/06_Working_with_Batch_and_Stochastic_Training)
  * Tensorflow makes it easy to use both batch and stochastic training. We show how to implement both and talk about the benefits and limitations of each.
 7. [Combining Everything Together](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/07_Combining_Everything_Together)
  * We now combine everything together that we have learned and create a simple classifier.
 8. [Evaluating Models](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/02_Tensorflow_Way/08_Evaluating_Models)
  * Any model is only as good as it's evaluation.  Here we show two examples of (1) evaluating a regression algorithm and (2) a classification algorithm.

## [Ch 3: Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression)

Here we show how to implement various linear regression techniques in Tensorflow.  The first two sections show how to do standard matrix linear regression solving in Tensorflow.  The remaining six sections depict how to implement various types of regression using computational graphs in Tensorflow.

 1. [Using the Matrix Inverse Method](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/01_Using_the_Matrix_Inverse_Method)
  * How to solve a 2D regression with a matrix inverse in Tensorflow.
 2. [Implementing a Decomposition Method](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/02_Implementing_a_Decomposition_Method)
  * Solving a 2D linear regression with Cholesky decomposition.
 3. [Learning the Tensorflow Way of Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/03_Tensorflow_Way_of_Linear_Regression)
  * Linear regression iterating through a computational graph with L2 Loss.
 4. [Understanding Loss Functions in Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/04_Loss_Functions_in_Linear_Regressions)
  * L2 vs L1 loss in linear regression.  We talk about the benefits and limitations of both.
 5. [Implementing Deming Regression (Total Regression)](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/05_Implementing_Deming_Regression)
  * Deming (total) regression implmented in Tensorflow by changing the loss function.
 6. [Implementing Lasso and Ridge Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression)
  * Lasso and Ridge regression are ways of regularizing the coefficients. We implement both of these in Tensorflow via changing the loss functions.
 7. [Implementing Elastic Net Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/07_Implementing_Elasticnet_Regression)
  * Elastic net is a regularization technique that combines the L2 and L1 loss for coefficients.  We show how to implement this in Tensorflow.
 8. [Implementing Logistic Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/03_Linear_Regression/08_Implementing_Logistic_Regression)
  * We implment logistic regression by the use of an activation function in our computational graph.

## [Ch 4: Support Vector Machines](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines)

This chapter shows how to implement various SVM methods with Tensorflow.  We first create a linear SVM and also show how it can be used for regression.  We then introduce kernels (RBF Gaussian kernel) and show how to use it to split up non-linear data. We finish with a multi-dimensional implementation of non-linear SVMs to work with multiple classes.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/01_Introduction)
  * We introduce the concept of SVMs and how we will go about implementing them in the Tensorflow framework.
 2. [Working with Linear SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/02_Working_with_Linear_SVMs)
  * We create a linear SVM to separate I. setosa based on sepal length and pedal width in the Iris data set.
 3. [Reduction to Linear Regression](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/03_Reduction_to_Linear_Regression)
  * The heart of SVMs is separating classes with a line.  We change tweek the algorithm slightly to perform SVM regression.
 4. [Working with Kernels in Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/04_Working_with_Kernels)
  * In order to extend SVMs into non-linear data, we explain and show how to implement different kernels in Tensorflow.
 5. [Implmenting Non-Linear SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/05_Implementing_Nonlinear_SVMs)
  * We use the Gaussian kernel (RBF) to separate non-linear classes.
 6. [Implementing Multi-class SVMs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs)
  * SVMs are inherently binary predictors.  We show how to extend them in a one-vs-all strategy in Tensorflow.

## [Ch 5: Nearest Neighbor Methods](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods)

Nearest Neighbor methods are a very popular ML algorithm.  We show how to implement k-Nearest Neighbors, weighted k-Nearest Neighbors, and k-Nearest Neighbors with mixed distance functions.  In this chapter we also show how to use the Levenshtein distance (edit distance) in Tensorflow, and use it to calculate the distance between strings. We end this chapter with showing how to use k-Nearest Neighbors for categorical prediction with the MNIST handwritten digit recognition.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/01_Introduction)
  * We introduce the concepts and methods needed for performing k-Nearest Neighbors in Tensorflow.
 2. [Working with Nearest Neighbors](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/02_Working_with_Nearest_Neighbors)
  * We create a nearest neighbor algorithm that tries to predict housing worth (regression).
 3. [Working with Text Based Distances](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/03_Working_with_Text_Distances)
  * In order to use a distance function on text, we show how to use edit distances in Tensorflow.
 4. [Computing Mixing Distance Functions](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions)
  * Here we implement scaling of the distance function by the standard deviation of the input feature for k-Nearest Neighbors.
 5. [Using Address Matching](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example)
  * We use a mixed distance function to match addresses. We use numerical distance for zip codes, and string edit distance for street names. The street names are allowed to have typos.
 6. [Using Nearest Neighbors for Image Recognition](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition)
  * The MNIST digit image collection is a great data set for illustration of how to perform k-Nearest Neighbors for an image classification task.

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
  * Given a set of tic-tac-toe boards and corresponding optimal moves, we train a neural network classification model to play.  At the end of the script, you can attempt to play against the trained model.

## [Ch 7: Natural Language Processing](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing)

Natural Language Processing (NLP) is a way of processing textual information into numerical summaries, features, or models. In this chapter we will motivate and explain how to best deal with text in Tensorflow.  We show how to implement the classic 'Bag-of-Words' and show that there may be better ways to embed text based on the problem at hand. There are neural network embeddings called Word2Vec (CBOW and Skip-Gram) and Doc2Vec.  We show how to implement all of these in Tensorflow.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/01_Introduction)
  * We introduce methods for turning text into numerical vectors. We introduce the Tensorflow 'embedding' feature as well.
 2. [Working with Bag-of-Words](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/02_Working_with_Bag_of_Words)
  * Here we use Tensorflow to do a one-hot-encoding of words called bag-of-words.  We use this method and logistic regression to predict if a text message is spam or ham.
 3. [Implementing TF-IDF](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/03_Implementing_tf_idf)
  * We implement Text Frequency - Inverse Document Frequency (TFIDF) with a combination of Sci-kit Learn and Tensorflow. We perform logistic regression on TFIDF vectors to improve on our spam/ham text-message predictions.
 4. [Working with CBOW](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/04_Working_With_Skip_Gram_Embeddings)
  * Our first implementation of Word2Vec called, "skip-gram" on a movie review database.
 5. [Working with Skip-Gram](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/05_Working_With_CBOW_Embeddings)
  * Next, we implement a form of Word2Vec called, "CBOW" (Continuous Bag of Words) on a movie review database.  We also introduce method to saving and loading word embeddings.
 6. [Implementing Word2Vec Example](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/06_Using_Word2Vec_Embeddings)
  * In this example, we use the prior saved CBOW word embeddings to improve on our TF-IDF logistic regression of movie review sentiment.
 7. [Performing Sentiment Analysis with Doc2Vec](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/07_Natural_Language_Processing/07_Sentiment_Analysis_With_Doc2Vec)
  * Here, we introduce a Doc2Vec method (concatenation of doc and word emebeddings) to improve out logistic model of movie review sentiment.

## [Ch 8: Convolutional Neural Networks](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks)

Convolutional Neural Networks (CNNs) are ways of getting neural networks to deal with image data. CNN derive their name from the use of a convolutional layer that applies a fixed size filter across a larger image, recognizing a pattern in any part of the image. There are many other tools that they use (max-pooling, dropout, etc...) that we show how to implement with Tensorflow.  We also show how to retrain an existing architecture and take CNNs further with Stylenet and Deep Dream.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/01_Intro_to_CNN)
  * We introduce convolutional neural networks (CNN), and how we can use them in Tensorflow.
 2. [Implementing a Simple CNN.](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/02_Intro_to_CNN_MNIST)
  * Here, we show how to create a CNN architecture that performs well on the MNIST digit recognition task.
 3. [Implementing an Advanced CNN.](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/03_CNN_CIFAR10)
  * In this example, we show how to replicate an architecture for the CIFAR-10 image recognition task.
 4. [Retraining an Existing Architecture.](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/04_Retraining_Current_Architectures)
  * We show how to download and setup the CIFAR-10 data for the Tensorflow retraining/fine-tuning tutorial.
 5. [Using Stylenet/NeuralStyle.](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/05_Stylenet_NeuralStyle)
  * In this recipe, we show a basic implementation of using Stylenet or Neuralstyle.
 6. [Implementing Deep Dream.](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/08_Convolutional_Neural_Networks/06_Deepdream)
  * This script shows a line-by-line explanation of Tensorflow's deepdream tutorial. Taken from [Deepdream on Tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream). Note that the code here is converted to Python 3.

## [Ch 9: Recurrent Neural Networks](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks)

Recurrent Neural Networks (RNNs) are very similar to regular neural networks except that they allow 'recurrent' connections, or loops that depend on the prior states of the network. This allows RNNs to efficiently deal with sequential data, whereas other types of networks cannot. We then motivate the usage of LSTM (Long Short Term Memory) networks as a way of addressing regular RNN problems. Then we show how easy it is to implement these RNN types in Tensorflow.

 1. [Introduction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/01_Introduction)
  * We introduce Recurrent Neural Networks and how they are able to feed in a sequence and predict either a fixed target (categorical/numerical) or another sequence (sequence to sequence).
 2. [Implmenting an RNN Model for Spam Prediction](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/02_Implementing_RNN_for_Spam_Prediction)
  * In this example, we create an RNN model to improve on our spam/ham SMS text predictions.
 3. [Imlementing an LSTM Model for Text Generation](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/03_Implementing_LSTM)
  * We show how to implement a LSTM (Long Short Term Memory) RNN for Shakespeare language generation. (Word level vocabulary)
 4. [Stacking Multiple LSTM Layers](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/04_Stacking_Multiple_LSTM_Layers)
  * We stack multiple LSTM layers to improve on our Shakespeare language generation. (Character level vocabulary)
 5. [Creating a Sequence to Sequence Translation Model (Seq2Seq)](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/09_Recurrent_Neural_Networks/05_Creating_A_Sequence_To_Sequence_Model)
  * Here, we use Tensorflow's sequence-to-sequence models to train an English-German translation model.

## [Ch 10: Taking Tensorflow to Production](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production)

 1. [Implementing Unit Tests](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production/01_Implementing_Unit_Tests)
  * We show how to implement different types of unit tests on tensors (placeholders and variables).
 2. [Using Multiple Executors (Devices)](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production/02_Using_Multiple_Devices)
  * How to use a machine with multiple devices.  E.g., a machine with a CPU, and one or more GPUs.
 3. [Parallelizing Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production/03_Parallelizing_Tensorflow)
  * How to setup and use Tensorflow distributed on multiple machines.
 4. [Tips for Tensorflow in Production](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production/04_Production_Tips)
  * Various tips for devloping with Tensorflow
 5. [An Example of Productionalizing Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_Tensorflow_to_Production/05_Production_Example)
  * We show how to do take the RNN model for predicting ham/spam (from Chapter 9, recipe #2) and put it in two production level files: training and evaluation.

## [Ch 11: More with Tensorflow](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/11_More_with_Tensorflow)

 1. [Visualizing Computational Graphs (with Tensorboard)](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/11_More_with_Tensorflow/01_Visualizing_Computational_Graphs)
  * An example of using histograms, scalar summaries, and creating images in Tensorboard.
 2. [Clustering Using K-means](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/11_More_with_Tensorflow/03_Clustering_Using_KMeans)
  * How to use Tensorflow to do k-means clustering.  We use the Iris data set, set k=3, and use k-means to make predictions.
 3. [Working with a Genetic Algorithm](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/11_More_with_Tensorflow/02_Working_with_a_Genetic_Algorithm)
  * We create a genetic algorithm to optimize an individual (array of 50 numbers) toward the ground truth function.
 4. [Solving a System of ODEs](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/11_More_with_Tensorflow/04_Solving_A_System_of_ODEs)
  * Here, we show how to use Tensorflow to solve a system of ODEs.  The system of concern is the Lotka-Volterra predator-prey system.

