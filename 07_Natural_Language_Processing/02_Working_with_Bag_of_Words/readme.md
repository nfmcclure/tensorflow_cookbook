# Working with Bag of Words

To illustrate how to use bag of words with a text data set, we will use a spam-ham phone text database from the UCI machine learning data repository (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).  This is a collection of phone text messages that are spam or not-spam (ham).

We will download this data, store it for future use, and then proceed with the bag of words method to predict if a text is spam or not.  The model that will operate on the bag of words will be a logistic model with no hidden layer.  We will use stochastic training, with batch size of one, and compute the accuracy on a held out test set at the end.