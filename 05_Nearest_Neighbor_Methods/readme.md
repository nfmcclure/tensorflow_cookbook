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
