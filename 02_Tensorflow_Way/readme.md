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
