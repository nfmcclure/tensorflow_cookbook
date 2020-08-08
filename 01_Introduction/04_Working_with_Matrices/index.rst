.. important::

   理解TensorFlow如何处理矩阵对于理解计算图中的数据流动是很重要的。
   
   很多算法都依赖与矩阵运算。TensorFlow可以给我们一个简单操作来完成矩阵运算。对于下面所有的例子，我们通过运行下面的命令都先建立一个 :code:`graph session` :

.. code:: python
   :number-lines: 1
   
   >>> import tensorflow as tf
   >>> sess = tf.compat.v1.Session()
   >>> from tensorflow.python.framework import ops
   >>> ops.reset_default_graph()
   >>> tf.compat.v1.disable_eager_execution()
   
创建一个矩阵 
-----------

我们可以通过 :code:`numpy` 数组或者嵌套列表来创建一个二维矩阵，就像我们在张量那一节所描述的那样 ( :code:`convert_to_tensor` )。我们也可以使用张量创建函数并为这些函数( :code:`zeros()ones()truncated_normal()` 等等)设定一个二维的形状(因为矩阵就是二维张量)。 TensorFlow也允许我们用 :code:`diag()` 从一维数组或者列表中创建一个对角矩阵。例如：

.. code:: python
   
   # 对角矩阵
   >>> identiy_matrix = tf.compat.v1.diag([1.0, 1.0, 1.0])
   >>> print(sess.run(identiy_matrix))
   [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]
   # 创建一个二维随机张量，也就是随机矩阵
   >>> A = tf.compat.v1.truncated_normal([2,3])
   >>> print(sess.run(A))
   [[ 0.19759183 -1.436814   -1.107715  ]
    [-0.6905967  -0.19711868  0.6596967 ]]
   # 创建一个二维常数填充张量，也就是常数矩阵
   >>> B = tf.fill([2,3],5.0)
   >>> print(sess.run(B))
   [[5. 5. 5.]
    [5. 5. 5.]]
   # 创建一个二维随机张量，也就是随机矩阵
   >>> C = tf.compat.v1.random_uniform([3,2])
   >>> print(sess.run(C))
   [[0.3477279  0.39023817]
    [0.38307    0.8967395 ]
    [0.8217212  0.32184577]]
    # 使用内置函数convert_to_tensor将数组转化成张量
   >>> D = tf.compat.v1.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
   >>> print(sess.run(D))
   [[ 1.  2.  3.]
    [-3. -7. -1.]
    [ 0.  5. -2.]]
   # 非传统意义上的矩阵
   >>> E = tf.zeros([2,3,3])
   >>> print(sess.run(E))
   [[[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]

    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]]
   
矩阵加减法 
-----------

.. code:: python
   
   # 加法
   >>> print(sess.run(A+B))
   [[4.2034802 5.6497774 6.104109 ]
    [3.8710573 5.6505775 4.063135 ]]
   # 减法
   >>> print(sess.run(B-B))
   [[0. 0. 0.]
    [0. 0. 0.]]
   # 乘法
   >>> print(sess.run(tf.matmul(B, identiy_matrix)))
   [[5. 5. 5.]
    [5. 5. 5.]]
   # 矩阵运算需要注意两个的维度，否则容易出错
   >>> print(sess.run(tf.matmul(A, B)))
   Traceback (most recent call last):
   ...
   ValueError: Dimensions must be equal
   # 如果对某个模块不明白，可以调用help函数
   >>> help(tf.matmul)
   Help on function matmul in module tensorflow.python.ops.math_ops:
   ...
   matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
   # 矩阵的转置
   >>> print(sess.run(tf.transpose(C)))
   [[0.11786842 0.32758367 0.54398596]
    [0.35542393 0.546188   0.6743456 ]]
   # 对于行列式，可以用
   >>> print(sess.run(tf.compat.v1.matrix_determinant(D)))
   -37.99999999999999
   # 矩阵的逆(inverse)
   # 注意，如果矩阵是对称正定矩阵，则矩阵的逆是基于Cholesky分解，否则基于LU分解。
   >>> print(sess.run(tf.compat.v1.matrix_inverse(D)))
   [[-0.5        -0.5        -0.5       ]
    [ 0.15789474  0.05263158  0.21052632]
    [ 0.39473684  0.13157895  0.02631579]]
   # 对于矩阵的本征值和本征向量，用下面的代码
   >>> print(sess.run(tf.compat.v1.self_adjoint_eigvals(D)))
   [-10.65907521  -0.22750691   2.88658212]
   # self_adjoint_eig()输出一个数组是本征值，输出第二数组为本征向量, 这在数学上叫本征分解
   >>> print(sess.run(tf.compat.v1.self_adjoint_eig(D)[0]))
   [-10.65907521  -0.22750691   2.88658212]
   >>> print(sess.run(tf.compat.v1.self_adjoint_eig(D)[1]))
   [[ 0.21749542  0.63250104 -0.74339638]
    [ 0.84526515  0.2587998   0.46749277]
    [-0.4880805   0.73004459  0.47834331]]


Start a graph session

.. code:: python
    
    sess = tf.Session()

Declaring matrices
^^^^^^^^^^^^^^^^^^

Identity Matrix:

.. code:: python

  identity_matrix = tf.diag([1.0,1.0,1.0])
  print(sess.run(identity_matrix))

the output::

  [[ 1.  0.  0.]
  [ 0.  1.  0.]
  [ 0.  0.  1.]]
  
  
2x3 random norm matrix:

.. code:: python

  A = tf.truncated_normal([2,3])
  print(sess.run(A))

the output::

  [[-0.09611617  1.50501597  0.42943364]
  [ 0.04031758 -0.66115439 -0.91324311]]

2x3 constant matrix:

.. code:: python

  B = tf.fill([2,3], 5.0)
  print(sess.run(B))

the output::

  [[ 5.  5.  5.]
  [ 5.  5.  5.]]

3x2 random uniform matrix:

.. code:: python

  C = tf.random_uniform([3,2])
  print(sess.run(C))

the output::

  [[ 0.34232175  0.16590214]
  [ 0.70915234  0.25312507]
  [ 0.11254978  0.03158247]]

Create matrix from np array:

.. code:: python
  
  D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
  print(sess.run(D))

the output::

  [[ 1.  2.  3.]
  [-3. -7. -1.]
  [ 0.  5. -2.]]

Matrix Operations
^^^^^^^^^^^^^^^^^^

Matrix addition/subtraction:

.. code:: python

  print(sess.run(A+B))
  print(sess.run(B-B))
  
the output::

  [[ 3.69020724  5.68584728  4.3044405 ]
  [ 6.57195997  3.92733717  5.5748148 ]]
  [[ 0.  0.  0.]
  [ 0.  0.  0.]]
  
Matrix Multiplication:

.. code:: python

  print(sess.run(tf.matmul(B, identity_matrix)))

the output::

  [[ 5.  5.  5.]
  [ 5.  5.  5.]]
  
Matrix Transpose:

.. code:: python

  print(sess.run(tf.transpose(C)))
  
  
the output::

  [[ 0.11936677  0.07210469  0.06045544]
  [ 0.93742907  0.29088366  0.43557048]]


Matrix Determinant:

.. code:: python

  print(sess.run(tf.matrix_determinant(D)))

the output::

  -38.0
  
  
Matrix Inverse:

.. code:: python

  print(sess.run(tf.matrix_inverse(D)))
  
the output::

  [[-0.5        -0.5        -0.5       ]
  [ 0.15789474  0.05263158  0.21052632]
  [ 0.39473684  0.13157895  0.02631579]]


Cholesky Decomposition:

.. code:: python

  print(sess.run(tf.cholesky(identity_matrix)))

the output::

  [[ 1.  0.  0.]
  [ 0.  1.  0.]
  [ 0.  0.  1.]]
  
Eigenvalues and Eigenvectors: We use `tf.self_adjoint_eig()` function, which returns two objects, first one 
is an array of eigenvalues, the second is a matrix of the eigenvectors.

.. code:: python

  eigenvalues, eigenvectors = sess.run(tf.self_adjoint_eig(D))
  print(eigenvalues)
  print(eigenvectors)
  
 the output::
 
  [-10.65907521  -0.22750691   2.88658212]
  [[ 0.21749542  0.63250104 -0.74339638]
  [ 0.84526515  0.2587998   0.46749277]
  [-0.4880805   0.73004459  0.47834331]]
