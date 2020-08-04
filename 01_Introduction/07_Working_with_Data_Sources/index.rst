This recipe shows how we will be accessing the datasets necessary for the rest of the 
book.

We start by loading the necessary libraries and resetting the computational graph.

.. code:: python

  import matplotlib.pyplot as plt
  import numpy as np
  import tensorflow as tf
  from tensorflow.python.framework import ops
  ops.reset_default_graph()
  
The Iris Dataset (R. Fisher / Scikit-Learn)
--------------------------------------------

One of the most frequently used ML datasets is the iris flower dataset. We will use 
the easy import tool, datasets from scikit-learn. You can read more about it `here <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris>`_

.. code:: python

  from sklearn.datasets import load_iris

  iris = load_iris()
  print(len(iris.data))
  print(len(iris.target))
  print(iris.data[0])
  print(set(iris.target))

the output::

  150
  150
  [ 5.1  3.5  1.4  0.2]
  {0, 1, 2}
  
Low Birthrate Dataset (Hosted on Github)
----------------------------------------
The 'Low Birthrate Dataset' is a dataset from a famous study by Hosmer and Lemeshow 
in 1989 called, "Low Infant Birth Weight Risk Factor Study". It is a very commonly 
used academic dataset mostly for logistic regression. We will host this dataset on the
public Github `here <https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat>`_


.. code:: python
  
  import requests
  birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
  birth_file = requests.get(birthdata_url)
  birth_data = birth_file.text.split('\r\n')
  birth_header = birth_data[0].split('\t')
  birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
  print(len(birth_data))
  print(len(birth_data[0]))

the output::

  189
  9
  
Housing Price Dataset (UCI)
-----------------------------
We will also use a housing price dataset from the University of California at Irvine 
(UCI) Machine Learning Database Repository. It is a great regression dataset to use. 
You can read more about it `here <https://archive.ics.uci.edu/ml/datasets/Housing>`_


.. code:: python

  import requests
  housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
  housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
  housing_file = requests.get(housing_url)
  housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
  print(len(housing_data))
  print(len(housing_data[0]))
  
the output::

  506
  14
  
MNIST Handwriting Dataset (Yann LeCun)
-------------------------------------
The MNIST Handwritten digit picture dataset is the Hello World of image recognition. 
The famous scientist and researcher, Yann LeCun, hosts it on his webpage `here <http://yann.lecun.com/exdb/mnist/>`_. 
But because it is so commonly used, many libraries, including TensorFlow, host it 
internally. We will use TensorFlow to access this data as follows.

If you haven't downloaded this before, please wait a bit while it downloads

.. code:: python

  from tensorflow.examples.tutorials.mnist import input_data
  
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  print(len(mnist.train.images))
  print(len(mnist.test.images))
  print(len(mnist.validation.images))
  print(mnist.train.labels[1,:])

the output::

  Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
  Extracting MNIST_data/train-images-idx3-ubyte.gz
  Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
  Extracting MNIST_data/train-labels-idx1-ubyte.gz
  Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
  Extracting MNIST_data/t10k-images-idx3-ubyte.gz
  Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
  Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
  55000
  10000
  5000
  [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]

CIFAR-10 Data
--------------
The `CIFAR-10 data <https://www.cs.toronto.edu/~kriz/cifar.html>`_ contains 60,000 
32x32 color images of 10 classes collected by Alex Krizhevsky, Vinod Nair, and 
Geoffrey Hinton. Alex Krizhevsky maintains the page referenced here. This is such a
common dataset, that there are built in functions in TensorFlow to access this data 
(the keras wrapper has these commands). Note that the keras wrapper for these functions
automatically splits the images into a 50,000 training set and a 10,000 test set.

.. code:: python

  from PIL import Image
  # Running this command requires an internet connection and a few minutes to download all the images.
  (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

the output:: 

  Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  The ten categories are (in order):
  
  Airplane
  Automobile
  Bird
  Car
  Deer
  Dog
  Frog
  Horse
  Ship
  Truck

.. code:: python
  
  X_train.shape
  y_train.shape
  y_train[0,] # this is a frog
  # Plot the 0-th image (a frog)
  %matplotlib inline
  img = Image.fromarray(X_train[0,:,:,:])
  plt.imshow(img)

the output::

  (50000, 32, 32, 3)
  (50000, 1)
  array([6], dtype=uint8)
  <matplotlib.image.AxesImage at 0x7ffb48a47400>

Ham/Spam Texts Dataset (UCI)
----------------------------

We will use another UCI ML Repository dataset called the SMS Spam Collection. You can 
read about it `here <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`. 
As a sidenote about common terms, when predicting if a data point represents 'spam' 
(or unwanted advertisement), the alternative is called 'ham' (or useful information).

This is a great dataset for predicting a binary outcome (spam/ham) from a textual input.
This will be very useful for short text sequences for Natural Language Processing 
(Ch 7) and Recurrent Neural Networks (Ch 9).

.. code:: python

  import requests
  import io
  from zipfile import ZipFile

  # Get/read zip file
  zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
  r = requests.get(zip_url)
  z = ZipFile(io.BytesIO(r.content))
  file = z.read('SMSSpamCollection')
  # Format Data
  text_data = file.decode()
  text_data = text_data.encode('ascii',errors='ignore')
  text_data = text_data.decode().split('\n')
  text_data = [x.split('\t') for x in text_data if len(x)>=1]
  [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
  print(len(text_data_train))
  print(set(text_data_target))
  print(text_data_train[1])

the output::

  5574
  {'spam', 'ham'}
  Ok lar... Joking wif u oni...
  
  
Movie Review Data (Cornell)
---------------------------
The Movie Review database, collected by Bo Pang and Lillian Lee (researchers at Cornell),
serves as a great dataset to use for predicting a numerical number from textual inputs.


You can read more about the dataset and papers using it `here <https://www.cs.cornell.edu/people/pabo/movie-review-data/>`

.. code:: python

  import requests
  import io
  import tarfile

  movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
  r = requests.get(movie_data_url)
  # Stream data into temp object
  stream_data = io.BytesIO(r.content)
  tmp = io.BytesIO()
  while True:
      s = stream_data.read(16384)
      if not s:  
         break
      tmp.write(s)
  stream_data.close()
  tmp.seek(0)
  # Extract tar file
  tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
  pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
  neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
  # Save pos/neg reviews
  pos_data = []
  for line in pos:
      pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  neg_data = []
  for line in neg:
      neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  tar_file.close()
  
  print(len(pos_data))
  print(len(neg_data))
  print(neg_data[0])
  
the output::

  5331
  5331
  simplistic , silly and tedious . 

The Complete Works of William Shakespeare (Gutenberg Project)
-------------------------------------------------------------
For training a TensorFlow Model to create text, we will train it on the complete works
of William Shakespeare. This can be accessed through the good work of the Gutenberg 
Project. The Gutenberg Project frees many non-copyright books by making them accessible
for free from the hard work of volunteers.

You can read more about the Shakespeare works `here <http://www.gutenberg.org/ebooks/100>`_

.. code:: python

  # The Works of Shakespeare Data
  import requests

  shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
  # Get Shakespeare text
  response = requests.get(shakespeare_url)
  shakespeare_file = response.content
  # Decode binary into string
  shakespeare_text = shakespeare_file.decode('utf-8')
  # Drop first few descriptive paragraphs.
  shakespeare_text = shakespeare_text[7675:]
  print(len(shakespeare_text))

the output::

  5582212
  
English-German Sentence Translation Database (Manythings/Tatoeba)
-----------------------------------------------------------------

The `Tatoeba Project <http://www.manythings.org/corpus/about.html#info>` is also run 
by volunteers and is set to make the most bilingual sentence translations available 
between many different languages. Manythings.org compiles the data and makes it 
accessible.



`More bilingual sentence pairs <http://www.manythings.org/bilingual/>`_

.. code:: python

  # English-German Sentence Translation Data
  import requests
  import io
  from zipfile import ZipFile
  sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
  r = requests.get(sentence_url)
  z = ZipFile(io.BytesIO(r.content))
  file = z.read('deu.txt')
  # Format Data
  eng_ger_data = file.decode()
  eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
  eng_ger_data = eng_ger_data.decode().split('\n')
  eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
  [english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
  print(len(english_sentence))
  print(len(german_sentence))
  print(eng_ger_data[10])

the output::

  147788
  147788
  ['I won!', 'Ich hab gewonnen!']
