## Ch 5: Nearest Neighbor Methods

Nearest Neighbor methods are a very popular ML algorithm.  We show how to implement k-Nearest Neighbors, weighted k-Nearest Neighbors, and k-Nearest Neighbors with mixed distance functions.  In this chapter we also show how to use the Levenshtein distance (edit distance) in TensorFlow, and use it to calculate the distance between strings. We end this chapter with showing how to use k-Nearest Neighbors for categorical prediction with the MNIST handwritten digit recognition.

 1. [Introduction](01_Introduction#nearest-neighbor-methods-introduction)
  * We introduce the concepts and methods needed for performing k-Nearest Neighbors in TensorFlow.
 2. [Working with Nearest Neighbors](02_Working_with_Nearest_Neighbors#working-with-nearest-neighbors)
  * We create a nearest neighbor algorithm that tries to predict housing worth (regression).
 3. [Working with Text Based Distances](03_Working_with_Text_Distances#working-with-text-distances)
  * In order to use a distance function on text, we show how to use edit distances in TensorFlow.
 4. [Computing Mixing Distance Functions](04_Computing_with_Mixed_Distance_Functions#computing-with-mixed-distance-functions)
  * Here we implement scaling of the distance function by the standard deviation of the input feature for k-Nearest Neighbors.
 5. [Using Address Matching](05_An_Address_Matching_Example#an-address-matching-example)
  * We use a mixed distance function to match addresses. We use numerical distance for zip codes, and string edit distance for street names. The street names are allowed to have typos.
 6. [Using Nearest Neighbors for Image Recognition](06_Nearest_Neighbors_for_Image_Recognition#nearest-neighbors-for-image-recognition)
  * The MNIST digit image collection is a great data set for illustration of how to perform k-Nearest Neighbors for an image classification task.
