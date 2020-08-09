现在我们必须知道可以加到TensorFlow计算图上的其他计算工具。
除了标准的算式运算外，TensorFlow提供给我们更多需要注意的运算符，我们应当在继续之前知道如何使用它们。同样，我们需要运行一下下面的命令来创建一个 :code:`graph session`:

.. code:: python

  >>> import numpy as np
  >>> import tensorflow as tf
  >>> from tensorflow.python.framework import ops
  >>> ops.reset_default_graph()
  >>> tf.compat.v1.disable_eager_execution()
  >>> sess = tf.compat.v1.Session()

TensorFlow对张量有标准的运算符：:code:`add()` , :code:`sub()` , :code:`mul()` , 和 :code:`div()` . 需要指出的是，这部分所有的运算除了特别说明外，都会输出element-wise式输出结果。

:code:`div()` 函数及其相关的函数
------------------------------

:code:`div()` 返回与输出结果类型相同的结果。这意味着如果输入的是整数的话，它返回 :emphasis:`the floor of the division` (是 :code:`Python 2` 的近亲)。为了产生 :code:`Python 3` 版本的结果，TensorFlow提供了 :code:`truediv()` 函数，如下：

.. code:: python
  
  >>> print(sess.run(tf.compat.v1.div(3,4)))
  0
  >>> print(sess.run(tf.compat.v1.truediv(3,4)))
  0.75
  >>> print(sess.run(tf.compat.v1.div(3.0,4)))
  0.75
  
如果我们浮点数然后希望做一个整数除法，我们可以用 :code:`floordiv()` 函数。 需要注意的是，我们仍然返回一个浮点数，但是已经被近似成最近邻的整数。如下：

.. code:: python
  
  >>> print(sess.run(tf.compat.v1.floordiv(3.0,4.0)))
  0.0


:code:`mod()` 函数
------------------------------

另外一个重要的函数就是 :code:`mod()` . 这个函数返回除法的余数。如下：

.. code:: python
    
   >>> print(sess.run(tf.compat.v1.mod(22,5)))
   2
   >>> print(sess.run(tf.compat.v1.mod(22.0,5)))
   2.0

:code:`cross()` 函数
------------------------------

两个张量的叉乘可以通过调用 :py:func:`tensorflow.compat.v1.cross` 函数来实现。记住，这里的叉乘只定义到俩个三维向量，所以它仅支持俩个三维向量。如下：

.. code:: python
  
   >>> print(sess.run(tf.compat.v1.cross([1.,2.,3.],[4.,5.,6.])))
   [-3.  6. -3.]
   >>> help(tf.compat.v1.cross)

以下是 :py:func:`help` 函数返回的结果：

.. py:function:: cross(a, b, name=None)

   Compute the pairwise cross product.
   
   `a` and `b` must be the same shape; they can either be simple 3-element vectors, or any shape where the innermost dimension is 3. In the latter case, each pair of corresponding 3-element vectors is cross-multiplied independently.

   :param a: Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`. A tensor containing 3-element vectors.
   :type a: Tensor
   :param b: Must have the same type as `a`. 
   :type b: Tensor
   :param name: A name for the operation (optional).
   :rtype: Tensor 
   :returns: Has the same type as `a`.
   
常用的数学函数列表
---------------

.. attention:: 所有这些函数都是element-wise式运行。

     +--------------------------------------------+---------------------------------------+
     | 常用数学函数                               | 描述                                  |
     +============================================+=======================================+
     | :code:`tensorflow.compat.v1.abs()`         | 输入张量的绝对值                      |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.ceil()`        | 输入张量的向上舍入函数                |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.cos()`         | 输入张量的Cosine函数                  |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.exp()`         | 输入张量的exp函数                     |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.floor()`       | 输入张量的向下舍入函数                |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.inv()`         | 输入张量的倒数(用不了)                |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.log()`         | 输入张量的自然对数                    |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.maximum()`     | 输入张量的最大值                      |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.minimum`       | 输入张量的最小值                      |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.negative()`    | 输入张量的负值                        |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.pow()`         | 第一张量上升到第二张量元素            |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.round()`       | 输入张量的近似                        |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.rsqrt()`       | 输入张量平方根的倒数                  |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.sign()`        | 输出 -1, 0, 或 1 取决输入张量的符号   |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.sin()`         | 输入张量的Sine函数                    |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.sqrt()`        | 输入张量的平方根                      |
     +--------------------------------------------+---------------------------------------+
     | :code:`tensorflow.compat.v1.square()`      | 输入张量的平方                        |
     +--------------------------------------------+---------------------------------------+


以下是这些常用数学函数的实例：

.. code:: python

    >>> A = tf.fill([3,3],1.0)
    >>> B = tf.constant([[-1.,-2.,-3.],[-4.,-5.,-6.],[-7.,-8.,-9.]])
    
    >>> sess.run(tf.compat.v1.abs(B))
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]], dtype=float32)
       
    >>> sess.run(tf.compat.v1.ceil(B))
    array([[-1., -2., -3.],
           [-4., -5., -6.],
           [-7., -8., -9.]], dtype=float32)
       
    >>> sess.run(tf.compat.v1.cos(B))
    array([[ 0.5403023 , -0.4161468 , -0.9899925 ],
           [-0.6536436 ,  0.28366217,  0.96017027],
           [ 0.75390226, -0.14550003, -0.91113025]], dtype=float32)
    
    >>> C = sess.run(tf.compat.v1.ceil(sess.run(tf.compat.v1.cos(B))))
    >>> C
    array([[ 1., -0., -0.],
           [-0.,  1.,  1.],
           [ 1., -0., -0.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.exp(B))
    array([[3.6787945e-01, 1.3533528e-01, 4.9787067e-02],
           [1.8315639e-02, 6.7379470e-03, 2.4787523e-03],
           [9.1188197e-04, 3.3546262e-04, 1.2340980e-04]], dtype=float32)
    
    >>> D = sess.run(tf.compat.v1.floor(sess.run(tf.compat.v1.cos(B))))
    >>> D
    array([[ 0., -1., -1.],
           [-1.,  0.,  0.],
           [ 0., -1., -1.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.log(A))
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.maximum(A,C))
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
           
    >>> sess.run(tf.compat.v1.minimum(A,C))
    array([[ 1., -0., -0.],
           [-0.,  1.,  1.],
           [ 1., -0., -0.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.negative(B))
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]], dtype=float32)
    
    # 平方
    >>> sess.run(tf.compat.v1.pow(B,2))
    array([[ 1.,  4.,  9.],
           [16., 25., 36.],
           [49., 64., 81.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.round(C))
    array([[ 1., -0., -0.],
           [-0.,  1.,  1.],
           [ 1., -0., -0.]], dtype=float32)
    
    # rsqrt是指reverse + square root, 即求平方根之后再求倒数
    >>> E = sess.run(tf.compat.v1.rsqrt(tf.compat.v1.exp(B)))
    >>> E
    array([[ 1.6487212,  2.7182817,  4.481689 ],
           [ 7.3890557, 12.182494 , 20.085537 ],
           [33.11545  , 54.598145 , 90.017136 ]], dtype=float32)
    
    >>> F = sess.run(tf.compat.v1.sqrt(tf.compat.v1.exp(B)))
    >>> F
    array([[0.60653067, 0.36787942, 0.22313015],
           [0.13533528, 0.082085  , 0.04978707],
           [0.03019738, 0.01831564, 0.011109  ]], dtype=float32)
    >>> sess.run(tf.compat.v1.multiply(E,F))
    array([[1.        , 0.99999994, 0.99999994],
           [0.99999994, 1.        , 1.        ],
           [1.        , 0.9999998 , 1.        ]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.sign(A))
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.sign(A))
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    >>> sess.run(tf.compat.v1.sign(B))
    array([[-1., -1., -1.],
           [-1., -1., -1.],
           [-1., -1., -1.]], dtype=float32)
    
    >>> sess.run(tf.compat.v1.sign(D))
    array([[ 0., -1., -1.],
           [-1.,  0.,  0.],
           [ 0., -1., -1.]], dtype=float32)
    
以下是 :py:function:`tensorflow.compat.v1.sign` 的详细介绍：


.. py:function:: sign(x,name=None)
  
  返回矩阵元素的符号。如果 :code:`x < 0` , 返回 -1; 如果 :code:`x==0` , 返回 0; 如果 :code:`x>0` , 返回 1. 对于复数而言， 如果 :code:`x!=0`, 返回 :code:`y=sign(x)=x/|x|`, 否则返回 :code:`0` 。
  
  :param x: 必须下面中个一种类型，bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
  :type x: Tensor
  :keyword name: 操作符的名称(可选)
  :return: 和 :code:`x` (张量)形状
  :rtype: Tensor


.. code:: python
  
  >>> sess.run(tf.compat.v1.sin(B))
  array([[-0.84147096, -0.9092974 , -0.14112   ],
         [ 0.7568025 ,  0.9589243 ,  0.2794155 ],
         [-0.6569866 , -0.98935825, -0.4121185 ]], dtype=float32)
         
  >>> sess.run(tf.compat.v1.sqrt(A))
  array([[0.99999994, 0.99999994, 0.99999994],
         [0.99999994, 0.99999994, 0.99999994],
         [0.99999994, 0.99999994, 1.        ]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.square(B))
  array([[ 1.,  4.,  9.],
         [16., 25., 36.],
         [49., 64., 81.]], dtype=float32)
  

特殊数学函数列表
---------------

这里还有一些值得注意的特殊数学函数，这些函数可能会在机器学习中出现，幸运的是，TensorFlow有一些内置函数可以调用。值得注意的是，这些函数除了特殊说明，都是元素式运行的。

+-------------------------------------------------------+--------------------------------------------------------+
| 特殊数学函数                                          | 描述                                                   |
+=======================================================+========================================================+
| :code:`tensorflow.compat.v1.digamma()`                | :code:`Psi` 函数，是 :code:`lgamma()` 函数的导数       |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.erf()`                    | 输入张量的高斯误差函数(元素式运行)                     |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.erfc()`                   | 输入张量的高斯误差补余函数                             |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.igamma()`                 | 下正则不完全伽玛函数                                   |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.igammac()`                | 上正则不完全伽玛函数                                   |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.lbeta()`                  | :code:`beta` 函数绝对值的自然对数                      |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.lgamma()`                 | :code:`gamma` 函数绝对值的自然对数                     |
+-------------------------------------------------------+--------------------------------------------------------+
| :code:`tensorflow.compat.v1.squared_difference()`     | 两个张量差值的平方                                     |
+-------------------------------------------------------+--------------------------------------------------------+

下面给出这些函数的实例，详情请看本节学习模块：

.. code:: python
  
  >>> sess.run(tf.compat.v1.digamma(E))
  array([[0.16705728, 0.8049263 , 1.384306  ],
         [1.9308087 , 2.4583962 , 2.9749    ],
         [3.4848251 , 3.9908142 , 4.494435  ]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.erf(B))
  array([[-0.8427007, -0.9953223, -0.999978 ],
         [-1.       , -1.       , -1.       ],
         [-1.       , -1.       , -1.       ]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.erfc(B))
  array([[1.8427007, 1.9953222, 1.999978 ],
         [2.       , 2.       , 2.       ],
         [2.       , 2.       , 2.       ]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.igamma(A,E))
  array([[0.8077043 , 0.93401194, 0.9886857 ],
         [0.999382  , 0.9999949 , 1.        ],
         [1.        , 1.        , 1.        ]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.igammac(A,E))
  array([[1.9229566e-01, 6.5988049e-02, 1.1314288e-02],
         [6.1797921e-04, 5.1192928e-06, 1.8921789e-09],
         [4.1508981e-15, 1.9423487e-24, 0.0000000e+00]], dtype=float32)
  
  >>> sess.run(tf.compat.v1.lbeta(E))
  array([  -7.5096974,  -40.50966  , -182.8869   ], dtype=float32)
  
  >>> sess.run(tf.compat.v1.lgamma(E))
  array([[-1.0544503e-01,  4.4946167e-01,  2.4283466e+00],
         [ 7.3192654e+00,  1.7949518e+01,  3.9594162e+01],
         [ 8.1960083e+01,  1.6271490e+02,  3.1372983e+02]], dtype=float32)
  
  # 最简单理解的函数
  >>> sess.run(tf.compat.v1.squared_difference(A,B))
  array([[  4.,   9.,  16.],
         [ 25.,  36.,  49.],
         [ 64.,  81., 100.]], dtype=float32)


知道哪些函数可以用，可以加到计算图上，对我们来说很重要。大多数情况下，我们只需要关注前面提到函数。我们也可以自己定义函数或者自己根据数学表达式利用前面提到的函数，如下::
  
  # tan()函数 tan(pi/4) = 1
  >>> from numpy import pi
  >>> print(sess.run(tf.compat.v1.div(tf.sin(pi/4.),tf.cos(pi/4.))))
  1.0

自定义函数
---------------

如果我们想在计算图中加一些没在表格中出现的函数，我们可以通过前面的函数来创建自己想要的函数。这里一个函数例子，我们可以加到我们的计算图中：

.. code:: python
    
    >>> def custom_polynomial(value):
    ...   return (tf.compat.v1.subtract(3*tf.compat.v1.square(value),value)+10)
    >>> print(sess.run(custom_polynomial(11)))
    362

这里我们创建了一个多项式函数：
:math:`f(x) = 3 \ast x^2-x+10`


本节学习模块
------------

.. attention:: tensorflow.compat.v1.div函数介绍

.. automodule:: tensorflow.compat.v1.div
   :members:
   :undoc-members:
   :show-inheritance:
  
.. attention:: tensorflow.compat.v1.truediv函数介绍

.. automodule:: tensorflow.compat.v1.truediv
   :members:
   :undoc-members:
   :show-inheritance:

.. attention:: tensorflow.compat.v1.floordiv函数介绍

.. automodule:: tensorflow.compat.v1.floordiv
   :members:
   :undoc-members:
   :show-inheritance:

.. attention:: tensorflow.compat.v1.mod函数介绍

.. automodule:: tensorflow.compat.v1.mod
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.cross函数介绍

.. automodule:: tensorflow.compat.v1.cross
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.pow函数介绍

.. automodule:: tensorflow.compat.v1.pow
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.rsqrt函数介绍

.. automodule:: tensorflow.compat.v1.rsqrt
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.digamma函数介绍

.. automodule:: tensorflow.compat.v1.digamma
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.erf函数介绍

.. automodule:: tensorflow.compat.v1.erf
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.erfc函数介绍

.. automodule:: tensorflow.compat.v1.erfc
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.igamma函数介绍

.. automodule:: tensorflow.compat.v1.igamma
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.igammac函数介绍

.. automodule:: tensorflow.compat.v1.igammac
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.lbeta函数介绍

.. automodule:: tensorflow.compat.v1.lbeta
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.lgamma函数介绍

.. automodule:: tensorflow.compat.v1.lgamma
   :members:
   :undoc-members:
   :show-inheritance:
   
.. attention:: tensorflow.compat.v1.squared_difference函数介绍

.. automodule:: tensorflow.compat.v1.squared_difference
   :members:
   :undoc-members:
   :show-inheritance:
