引言
============

We introduce Recurrent Neural Networks and how they are able to feed in a sequence and predict either 
a fixed target (categorical/numerical) or another sequence (sequence to sequence).

卷积神经网络模型用于垃圾信息检测
=============================================

We create an RNN model to improve on our spam/ham SMS text predictions.

LSTM模型用于文本生成
===============================================

We show how to implement a LSTM (Long Short Term Memory) RNN for Shakespeare language generation. 
(Word level vocabulary)

堆叠多层LSTM
===============================

We stack multiple LSTM layers to improve on our Shakespeare language generation. (Character level 
vocabulary)

创建段对段模型翻译 (Seq2Seq)
============================================================

We show how to use TensorFlow's sequence-to-sequence models to train an English-German translation model.

训练Siamese相似度测量
=======================================

Here, we implement a Siamese RNN to predict the similarity of addresses and use it for record matching. 
Using RNNs for record matching is very versatile, as we do not have a fixed set of target categories and 
can use the trained model to predict similarities across new addresses.
