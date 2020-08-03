|Documentation Status| |MIT License| |Python version| |Huawei Clodu| |TensorFlow| |today| 

-------------------

.. |Documentation Status| image:: https://readthedocs.org/projects/tensorflow-ml/badge/?version=latest
   :target: https://tensorflow-ml.readthedocs.io/zh/latest/?badge=latest
.. |MIT License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: http://choosealicense.com/licenses/mit/
.. |Python version| image:: https://img.shields.io/badge/python-3.7,3.8-brightgreen.svg
   :target: https://www.python.org/
.. |Huawei Clodu| image:: https://img.shields.io/badge/platform-huawei%20cloud-blue
   :target: https://auth.huaweicloud.com/authui/login.html?service=https%3A%2F%2Fconsole.huaweicloud.com%2Fconsole%2F%3Flocale%3Dzh-cn#/login
.. |TensorFlow| image:: https://img.shields.io/badge/tensorflow-1.8-brightgreen.svg
   :target: https://github.com/tensorflow/tensorflow

TensorFlow 机器学习 Cookbook (version : |version|)
============================

TensorFlow (i.e., TF)::
   在2015年的时候已经成为开源项目, 自从那之后它已经成为Github中starred最多的机器学习库. TensorFlow的受欢迎度主要归功于它能帮助程序员创造计算图(computational graphs), 自动微分 (automatic differentation) 和 可定制性 (customizability). 由于这些特性，TensorFlow是一个强有力的灵活性高的工具,  用于解决很多机器学习的问题. 

本教程阐述很多机器学习算法, 以及如何把它们应用到实际情况中, 以及如何诠释所得到的结果.

.. important::

   - :ref:`第一章: 从TensorFlow开始 (Getting Started) <label1>`, 介绍主要tensorflow的对象与概念. 我们介绍张量, 变量和占位符. 我们也会展示如何在tensorflow中使用矩阵和其他的数学操作. 在本章的末尾，我们会展示如何获取数据资源.
   
   - :ref:`第二章: TensorFlow方式 (TF Way) <label2>`, 阐述如何用多种方式将第一章中所有的算法成分关联成一个计算图并创造出一个简单的分类器. 在阐述的过程中, 我们会介绍计算图 (computational graphs), 损失函数 (loss functions), 反向传播 (back propagation), 以及训练数据.
   
   - :ref:`第三章: 线性回归 (Linear Regression) <label3>`, 本章着重强调如何使用tensorflow来探索不同的线性回归技巧, 比如Deming, lasso, ridge, elastic net 和 logistic regression. 我们会在计算图中展示如何应用它们.
   
   - :ref:`第四章: 支持向量机 (Support Vector Machine) <label4>`, 介绍支持向量机 (SVMs) 然后展示如何用tensorflow去运用线性SVMs, 非线性SVMs和多类SVMs.
   
   - :ref:`第五章: 最近邻方法 (NNM) <label5>`, 展示如何运用数值度量，文本度量和比例距离函数使用最近邻技巧. 我们使用最近邻技巧来完成地址记录匹配和从MNIST数据库中对手写数字进行分类.
   
   - :ref:`第六章: 神经网络 (Neural Networks) <label6>`, 介绍了从操作门 (operational gates) 和激活函数 (activation function) 的概念开始, 在tensorflow中如何运用神经网络. 然后我们展示一个很浅神经元然后展示如何建立不同类型的层. 在本章的末尾, 我们会教tensorflow通过神经网络的方法玩井字棋(tic-tac-toe).
   
   - :ref:`第七章: 自然语言处理 (NLP) <label7>`, 本章展示了运用tensorflow不同文本的处理方法. 我们会展示如何在文本处理中使用Bag of Words (BoW) 模型和TF-IDF (Term Frequency-Inverse Document Frequency) 模型. 我们然后会用CBOW (Continuous Bag of Words) 和Skip-Gram模型来介绍神经元完了文本表达, 然后运用这些技巧到Word2Vec和Doc2Vec上, 用于解决实际结果预测.
   
   - :ref:`第八章: 卷积神经网络 (CNN) <label8>`, 通过展示如何通过使用卷积神经网络 (convolutional neural networks) CNNs模型将神经网络运用到图像处理上. 我们诠释了如何为MNIST数字识别构建一个简单卷积神经网络模型, 然后在CIFAR-10任务中把它扩展到颜色识别. 我们也会展示如何把之前训练过得图像识别模型扩展到自定义任务当中. 在本章的末尾，我们会在tensorflow中解释 stylenet/neural style和deep-dream 算法.
   
   - :ref:`第九章: 递归神经网络 (RNN) <label9>`, 会展示如何在tensorflow中运用递归神经元(recurrent neural networks). 我们会展示如何进行垃圾文本预测, 然后将递归神经网络模型扩展到基于莎士比亚文本生成. 我们也会训练段对段模型 (sequence to sequence model), 用于德语英语的翻译. 在本章的末尾, 我们也会展示Siamese递归神经网络用于地址记录匹配的用法.
   
   - :ref:`第十章: TensorFlow的应用技巧 <label10>`, 本章将会给出将TensorFlow应用到开发环境中, 如何利用多过程设备(比如GPUs), 然后将TensorFlow分布在多个机器上.
   
   - :ref:`第十一章: TensorFlow的更多功能 <label11>`, 通过阐述如何运行k-means, genetic算法来展示TensorFlow的多面性, 解决系统的常微分方程. 我们也展示Tensorboard的多处使用, 以及如何显示计算图度量.

本书主要针对


.. admonition::
      
      在本书中，经常有很多类型的文本可以区分不同的类型的信息。比如，*We then set the* `batch_size` *variable*。还有一整块的代码一般都设置成以下形式：
      :code: python
            
            embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1.0,1.0))
            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data_ph)

  
.. _label1:
.. toctree::
   :maxdepth: 2
   :caption: 从TensorFlow开始 (Getting Started)
   
   01_Introduction/index

.. _label2:
.. toctree::
   :maxdepth: 2
   :caption: TensorFlow方式 (TensorFlow Way)
   
   02_TensorFlow_Way/index

.. _label3:
.. toctree::
   :maxdepth: 2
   :caption: 线性回归 (Linear Regression)
   
   03_Linear_Regression/index

.. _label4:
.. toctree::
   :maxdepth: 2
   :caption: 支持向量机(Support Vector Machines)
   
   04_Support_Vector_Machines/index

.. _label5:
.. toctree::
   :maxdepth: 2
   :caption: 最近邻法 (Nearest Neighbor Methods)
   
   05_Nearest_Neighbor_Methods/index

.. _label6:
.. toctree::
   :maxdepth: 2
   :caption: 神经元网络 (Neural Networks)
   
   06_Neural_Networks/index

.. _label7:
.. toctree::
   :maxdepth: 2
   :caption: 自然语言处理
   
   07_Natural_Language_Processing/index

.. _label8:
.. toctree::
   :maxdepth: 2
   :caption: 卷积神经网络
   
   08_Convolutional_Neural_Networks/index

.. _label9:
.. toctree::
   :maxdepth: 2
   :caption: 递归神经网络
   
   09_Recurrent_Neural_Networks/index

.. _label10:
.. toctree::
   :maxdepth: 2
   :caption: TensorFlow的应用技巧
   
   10_Taking_TensorFlow_to_Production/index

.. _label11:
.. toctree::
   :maxdepth: 2
   :caption: TensorFlow的更多功能
   
   11_More_with_TensorFlow/index


许可证(License)
==============

MIT许可证请参见 `MIT LICENSE <https://github.com/nickcafferry/tensorflow/blob/master/LICENSE>`_

API
===

.. automodule:: run
   :members:
