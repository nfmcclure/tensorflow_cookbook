Recurrent Neural Networks
==========================

Introduction
--------------

We introduce Recurrent Neural Networks and how they are able to feed in a sequence and predict either 
a fixed target (categorical/numerical) or another sequence (sequence to sequence).

Implementing an RNN Model for Spam Prediction
---------------------------------------------

We create an RNN model to improve on our spam/ham SMS text predictions.

Implementing an LSTM Model for Text Generation
-----------------------------------------------

We show how to implement a LSTM (Long Short Term Memory) RNN for Shakespeare language generation. 
(Word level vocabulary)

Stacking Multiple LSTM Layers
-----------------------------

We stack multiple LSTM layers to improve on our Shakespeare language generation. (Character level 
vocabulary)

Creating a Sequence to Sequence Translation Model (Seq2Seq)
-----------------------------------------------------------

We show how to use TensorFlow's sequence-to-sequence models to train an English-German translation model.

Training a Siamese Similarity Measure
-------------------------------------

Here, we implement a Siamese RNN to predict the similarity of addresses and use it for record matching. 
Using RNNs for record matching is very versatile, as we do not have a fixed set of target categories and 
can use the trained model to predict similarities across new addresses.
