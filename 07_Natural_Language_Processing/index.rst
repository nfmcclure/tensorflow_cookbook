.. note::
   
   
引言
----------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/01_Introduction/index

We introduce methods for turning text into numerical vectors. We introduce the TensorFlow 'embedding' feature
as well.

------------

词袋 (Bag of Words)
---------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/02_Implementing_an_Operational_Gate/index


Here we use TensorFlow to do a one-hot-encoding of words called bag-of-words.  We use this method and logistic regression to predict if a text message is spam or ham.


.. image:: 

下载本章 :download:`Jupyter Notebook </06_Neural_Networks/02_Implementing_an_Operational_Gate/02_gates.ipynb>`

-----

词频-逆文本频率 (TF-IDF)
--------------
.. toctree::
       :maxdepth: 3
       
       /06_Neural_Networks/03_Working_with_Activation_Functions/index


We implement Text Frequency - Inverse Document Frequency (TFIDF) with a combination of Sci-kit Learn and TensorFlow. 


.. image:: 

下载本章 :download:`Jupyter Notebook </06_Neural_Networks/03_Working_with_Activation_Functions/03_activation_functions.ipynb>`

-----------

运用Skip-Gram
----------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/index


Our first implementation of Word2Vec called, "skip-gram" on a movie review database.

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/04_mixed_distance_functions_knn.ipynb>`

-----------

CBOW (Continuous Bag fo Words)
-------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/index

Next, we implement a form of Word2Vec called, "CBOW" (Continuous Bag of Words) on a movie review database.  We also introduce method to saving and loading word embeddings.

This section introduces the convolution layer and the max-pool layer.  We show how to chain these together in a 1D and 2D example with fully connected layers as well.


下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/05_address_matching.ipynb>`

-------------

Word2Vec应用实例
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index

In this example, we use the prior saved CBOW word embeddings to improve on our TF-IDF logistic regression of movie review sentiment.

Here we show how to functionalize different layers and variables for a cleaner multi-layer neural network.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`

-----------

Doc2Vec情感分析 (Sentiment Analysis)
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index


Here, we introduce a Doc2Vec method (concatenation of doc and word embeddings) to improve out logistic model of
movie review sentiment.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`


神经网络学习井字棋
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index

Given a set of tic-tac-toe boards and corresponding optimal moves, we train a neural network classification
model to play.  At the end of the script, we can attempt to play against the trained model.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`



本章学习模块
-----------

.. Submodules
.. ----------

*tensorflow\.zeros* 
^^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.zeros
    :members:
    :undoc-members:
    :show-inheritance:

------

*tensorflow\.ones*
^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.ones
    :members:
    :undoc-members:
    :show-inheritance:

-------------
