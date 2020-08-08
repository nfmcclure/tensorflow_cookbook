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


.. code:: python
   
   >>> identiy_matrix = tf.compat.v1.diag([1.0, 1.0, 1.0])
   >>> print(sess.run(identiy_matrix))
   [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]
   >>> A = tf.compat.v1.truncated_normal([2,3])
   >>> print(sess.run(A))
   [[ 0.19759183 -1.436814   -1.107715  ]
    [-0.6905967  -0.19711868  0.6596967 ]]
   >>> B = tf.fill([2,3],5.0)
   >>> print(sess.run(B))
   [[5. 5. 5.]
    [5. 5. 5.]]
   >>> C = tf.compat.v1.random_uniform([3,2])
   >>> print(sess.run(C))
   [[0.3477279  0.39023817]
    [0.38307    0.8967395 ]
    [0.8217212  0.32184577]]
   >>> D = tf.compat.v1.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
   >>> print(sess.run(D))
   [[ 1.  2.  3.]
    [-3. -7. -1.]
    [ 0.  5. -2.]]
   

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
