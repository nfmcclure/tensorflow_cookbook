Introduction
============

We introduce methods for turning text into numerical vectors. We introduce the TensorFlow 'embedding' feature
as well.

Working with Bag-of-Words
=========================

Here we use TensorFlow to do a one-hot-encoding of words called bag-of-words.  We use this method and logistic
regression to predict if a text message is spam or ham.

Implementing TF-IDF
====================

We implement Text Frequency - Inverse Document Frequency (TFIDF) with a combination of Sci-kit Learn and TensorFlow. 
We perform logistic regression on TFIDF vectors to improve on our spam/ham text-message predictions.

Working with Skip-Gram
======================

Our first implementation of Word2Vec called, "skip-gram" on a movie review database.

Working with CBOW
==================

Next, we implement a form of Word2Vec called, "CBOW" (Continuous Bag of Words) on a movie review database.  We 
also introduce method to saving and loading word embeddings.

Implementing Word2Vec Example
=============================

In this example, we use the prior saved CBOW word embeddings to improve on our TF-IDF logistic regression of movie
review sentiment.

Performing Sentiment Analysis with Doc2Vec
==========================================

Here, we introduce a Doc2Vec method (concatenation of doc and word embeddings) to improve out logistic model of
movie review sentiment.


