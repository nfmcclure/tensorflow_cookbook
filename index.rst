Tensorflow 机器学习 Cookbook
============================

TensorFlow在2015年的时候已经成为开源项目，自从那之后它已经成为Github中starred最多的机器学习库。
TensorFlow的受欢迎度主要归功于它能帮助程序员创造computational graphs,automatic differentation 和 
customizability。 由于这些特性，TensorFlow是一个强有力的灵活性高的工具，用于解决很多机器学习的问题。

本教程阐述很多机器学习算法，以及如何把它们应用到实际情况中，以及如何诠释所得到的结果。

.. toctree::
   :maxdepth: 2
   :caption: 从TensorFlow开始 (Getting Started with TensorFlow)
   
   - TensorFlow如何工作
   - 变量和张量的声明
   - 使用占位符和变量
   - 矩阵
   - 操作符的声明
   - 载入激活函数
   - 数据资源
   - 资源库

.. toctree::
   :maxdepth: 2
   :caption: TensorFlow算法 (The TensorFlow Way)
   
   - Computational Graph
   - 分层嵌套操作
   - 多层操作
   - 载入损失函数
   - 载入向后传递
   - 随机和批量训练
   
.. toctree::
   :maxdepth: 2
   :caption: 线性回归 (Linear Regression)
   
   - 矩阵转置
   - 矩阵分解法
   - TensorFLow的线性回归
   - 线性回归的损失函数
   - Deming回归(全回归)
   - 套索(Lasso)回归和岭(Ridge)回归
   - 弹性网(Elastic Net)回归
   - 逻辑(Logistic)回归
   
.. toctree::
   :maxdepth: 2
   :caption: 支持向量机(Support Vector Machines)
   
   - 引言
   - 线性支持向量机
   - 回归线性回归
   - TensorFlow中的核
   - 非线性支持向量机
   - 多类支持向量机
   
.. toctree::
   :maxdepth: 2
   :caption: 最近邻法 (Nearest Neighbor Methods)
   
   - 引言
   - 最近邻法的使用
   - 文本距离函数
   - 计算混合距离函数
   - 地址匹配
   - 图像处理的近邻法
   
.. toctree::
   :maxdepth: 2
   :caption: 神经元网络 (Neural Networks)
   
   - 引言
   - Implementing Operational Gates
   - Working with Gates and Activation Functions
   - Implementing a One Layer Neural Network
   - Implementing Different Layers
   - Using Multi-layer Neural Networks
   - Improving Predictions of Linear Models
   - Learning to Play Tic-Tac-Toe
   
.. toctree::
   :maxdepth: 2
   :caption: Natural Language Processing
   
   - Introduction
   - Working with Bag-of-Words
   - Implementing TF-IDF
   - Working with Skip-Gram
   - Working with CBOW
   - Implementing Word2Vec Example 
   - Performing Sentiment Analysis with Doc2Vec
   
.. toctree::
   :maxdepth: 2
   :caption: Convolutional Neural Networks
   
   - Introduction
   - Implementing a Simple CNN
   - Implementing an Advanced CNN
   - Retraining an Existing Architecture
   - Using Stylenet/NeuralStyle
   - Implementing Deep Dream

.. toctree::
   :maxdepth: 2
   :caption: Recurrent Neural Networks
   
   - Introduction
   - Implementing an RNN Model for Spam Prediction
   - Implementing an LSTM Model for Text Generation
   - Stacking Multiple LSTM Layers
   - Creating a Sequence to Sequence Translation Model (Seq2Seq)
   - Training a Siamese Similarity Measure

.. toctree::
   :maxdepth: 2
   :caption: Taking TensorFlow to Production
   
   - Implementing Unit Tests
   - Using Multiple Executors (Devices)
   - Parallelizing TensorFlow
   - Tips for TensorFlow in Production
   - An Example of Productionalizing TensorFlow

.. toctree::
   :maxdepth: 2
   :caption: More with TensorFlow
   
   - Visualizing Computational Graphs (with Tensorboard)
   - Working with a Genetic Algorithm
   - Clustering Using K-means
   - Solving a System of ODEs
   - Using a Random Forest
   - Using TensorFlow with Keras
