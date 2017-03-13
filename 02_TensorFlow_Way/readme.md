## Ch 2: The TensorFlow Way

After we have established the basic objects and methods in TensorFlow, we now want to establish the components that make up TensorFlow algorithms.  We start by introducing computational graphs, and then move to loss functions and back propagation.  We end with creating a simple classifier and then show an example of evaluating regression and classification algorithms.

 1. [One Operation as a Computational Graph](01_Operations_as_a_Computational_Graph#operations-as-a-computational-graph)
  * We show how to create an operation on a computational graph and how to visualize it using Tensorboard.
 2. [Layering Nested Operations](02_Layering_Nested_Operations#multiple-operations-on-a-computational-graph)
  * We show how to create multiple operations on a computational graph and how to visualize them using Tensorboard.
 3. [Working with Multiple Layers](03_Working_with_Multiple_Layers#working-with-multiple-layers)
  * Here we extend the usage of the computational graph to create multiple layers and show how they appear in Tensorboard.
 4. [Implementing Loss Functions](04_Implementing_Loss_Functions#implementing-loss-functions)
  * In order to train a model, we must be able to evaluate how well it is doing. This is given by loss functions. We plot various loss functions and talk about the benefits and limitations of some.
 5. [Implementing Back Propagation](05_Implementing_Back_Propagation#implementing-back-propagation)
  * Here we show how to use loss functions to iterate through data and back propagate errors for regression and classification.
 6. [Working with Stochastic and Batch Training](06_Working_with_Batch_and_Stochastic_Training#working-with-batch-and-stochastic-training)
  * TensorFlow makes it easy to use both batch and stochastic training. We show how to implement both and talk about the benefits and limitations of each.
 7. [Combining Everything Together](07_Combining_Everything_Together#combining-everything-together)
  * We now combine everything together that we have learned and create a simple classifier.
 8. [Evaluating Models](08_Evaluating_Models#evaluating-models)
  * Any model is only as good as it's evaluation.  Here we show two examples of (1) evaluating a regression algorithm and (2) a classification algorithm.
