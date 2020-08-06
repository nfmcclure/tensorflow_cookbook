.. note::

   Nearest Neighbor methods are a very popular ML algorithm.  We show how to implement k-Nearest 
   Neighbors, weighted k-Nearest Neighbors, and k-Nearest Neighbors with mixed distance functions. 
   In this chapter we also show how to use the Levenshtein distance (edit distance) in TensorFlow, 
   and use it to calculate the distance between strings. We end this chapter with showing how to 
   use k-Nearest Neighbors for categorical prediction with the MNIST handwritten digit recognition.

引言
----------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/01_Introduction/index
 
We introduce the concepts and methods needed for performing k-Nearest Neighbors in TensorFlow.

------------

最近邻法的使用
---------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/02_Working_with_Nearest_Neighbors/index

We create a nearest neighbor algorithm that tries to predict housing worth (regression).


.. image:: 

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/02_Working_with_Nearest_Neighbors/02_nearest_neighbor.ipynb>`

-----

文本距离函数
--------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/03_Working_with_Text_Distances/index

In order to use a distance function on text, we show how to use edit distances in TensorFlow.

.. image:: 

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/03_Working_with_Text_Distances/03_text_distances.ipynb>`

-----------

计算混合距离函数
----------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/index

Here we implement scaling of the distance function by the standard deviation of the input 
feature for k-Nearest Neighbors.


下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/04_Computing_with_Mixed_Distance_Functions/04_mixed_distance_functions_knn.ipynb>`

-----------

地址匹配
-------------
.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/index


We use a mixed distance function to match addresses. We use numerical distance for zip codes,
and string edit distance for street names. The street names are allowed to have typos.


下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/05_An_Address_Matching_Example/05_address_matching.ipynb>`

-------------

图像处理的近邻法
-----------

.. toctree::
       :maxdepth: 3
       
       /05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/index


The MNIST digit image collection is a great data set for illustration of how to perform 
k-Nearest Neighbors for an image classification task.

.. image::

下载本章 :download:`Jupyter Notebook </05_Nearest_Neighbor_Methods/06_Nearest_Neighbors_for_Image_Recognition/06_image_recognition.ipynb>`

-----------

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
