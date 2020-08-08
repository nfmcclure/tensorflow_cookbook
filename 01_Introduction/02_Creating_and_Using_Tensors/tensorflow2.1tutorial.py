#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
try:
    from tensorflow.python.framework import ops
except:
    print("No module named 'tensorflow.python.framework'"+tf.__version__)
ops.reset_default_graph()


# In[2]:


print(tf.__version__)


# In[3]:


tf.compat.v1.disable_eager_execution()


# In[4]:


sess = tf.compat.v1.Session()


# In[5]:


my_tensor = tf.zeros([1,20])


# In[6]:


sess.run(my_tensor)


# In[7]:


row_dim, col_dim = 3, 5
zero_tsr = tf.zeros([row_dim, col_dim])
sess.run(zero_tsr)


# In[8]:


row_dim, col_dim = 6, 7
ones_tsr = tf.ones([row_dim, col_dim])
sess.run(ones_tsr)


# In[9]:


row_dim, col_dim = 6, 7
filled_tsr = tf.fill([row_dim, col_dim],42)
sess.run(filled_tsr)


# In[10]:


constant1_tsr = tf.constant([1,2,3])
print(sess.run(constant1_tsr))
constant2_tsr = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print(sess.run(constant2_tsr))


# In[11]:


zeros_similar=tf.zeros_like(constant1_tsr)
sess.run(zeros_similar)


# In[12]:


ones_similar = tf.ones_like(constant2_tsr)
sess.run(ones_similar)


# In[13]:


linear_tsr = tf.linspace(start=0.0,stop=100,num=11)
sess.run(linear_tsr)


# In[14]:


integer_seq_tsr = tf.range(start=6,limit=15,delta=3)
sess.run(integer_seq_tsr)


# In[15]:


row_dim, col_dim = 8, 8
randuif_tsr = tf.compat.v1.random_uniform([row_dim, col_dim], minval=0, maxval=1)
sess.run(randuif_tsr)


# In[16]:


row_dim, col_dim = 8, 8
randnorm_tsr = tf.compat.v1.random_normal([row_dim,col_dim], mean=0.0, stddev=1.0)
sess.run(randnorm_tsr)


# In[17]:


row_dim, col_dim = 8, 8
runcnomr_tsr = tf.compat.v1.truncated_normal([row_dim,col_dim],mean=0.0, stddev=1.0)
sess.run(runcnomr_tsr)


# In[18]:


shuffled_output = tf.compat.v1.random_shuffle(runcnomr_tsr)
sess.run(shuffled_output)


# In[19]:


cropped_output = tf.compat.v1.random_crop(runcnomr_tsr,[4,4])
sess.run(cropped_output)


# In[20]:


sess.run(runcnomr_tsr)


# In[21]:


import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
image_raw_data_jpg=tf.compat.v1.gfile.GFile("Huawei logo.jpg","rb").read()
with sess as session:
    img_data=tf.image.decode_jpeg(image_raw_data_jpg)
    plt.figure(1)
    print(session.run(img_data))
    plt.imshow(img_data.eval())


# In[22]:


cropped_image = tf.compat.v1.random_crop(img_data, [3, 1, 3])


# In[23]:


sess = tf.compat.v1.Session()
sess.run(cropped_image)


# *Variable*

# In[24]:


my_var = tf.Variable(tf.zeros([1,20]))
sess.run(my_var.initializer)
sess.run(my_var)


# In[25]:


my_var1 = tf.Variable(tf.zeros([2,3]))
sess = tf.compat.v1.Session()
initialize_op = tf.compat.v1.global_variables_initializer()
sess.run(initialize_op)


# In[26]:


import numpy as np
sess = tf.compat.v1.Session()
x = tf.compat.v1.placeholder(tf.float32,shape=[2,2])
y = tf.compat.v1.identity(x)
x_vals = np.random.rand(2,2)
sess.run(y, feed_dict={x: x_vals})


# In[27]:


sess.run(x,feed_dict={x: x_vals})


# In[28]:


sess = tf.compat.v1.Session()
first_var = tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)
# 取决于第一个变量
second_var = tf.Variable(tf.zeros_like(first_var))
sess.run(second_var.initializer)


# In[29]:


row_dim, col_dim = 2, 3
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
ones_var = tf.Variable(tf.ones([row_dim,col_dim]))
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
print(sess.run(zero_var))
print(sess.run(ones_var))


# In[30]:


zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))
sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)
print(sess.run(ones_similar))
print(sess.run(zero_similar))


# In[31]:


fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
sess.run(fill_var.initializer)
print(sess.run(fill_var))


# In[32]:


# 通过常数列表来创建张量
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# 通过常数数组来创建变量张量
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))
   
sess.run(const_var.initializer)
sess.run(const_fill_var.initializer)

print(sess.run(const_var))

print(sess.run(const_fill_var))


# In[33]:


# TensorFlow的中linspace
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) 
# Generates [0.0, 0.5, 1.0] includes the end

# TensorFlow的range
sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) 
# Generates [6, 9, 12] doesn't include the end

sess.run(linear_var.initializer)
sess.run(sequence_var.initializer)

print(sess.run(linear_var))
print(sess.run(sequence_var))


# In[34]:


rnorm_var = tf.compat.v1.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
runif_var = tf.compat.v1.random_uniform([row_dim, col_dim], minval=0, maxval=4)
print(sess.run(rnorm_var))
print(sess.run(runif_var))


# In[ ]:


# 重设计算图
ops.reset_default_graph()

# 开始一个graph session
sess = tf.compat.v1.Session()

# 创建变量张量
my_var = tf.Variable(tf.zeros([1,20]))

# 将summary加到Tensorboard上
merged = tf.compat.v1.summary.merge_all()

# 初始化图形写入
writer = tf.compat.v1.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# 全局变量初始器
initialize_op = tf.compat.v1.global_variables_initializer()

# 变量初始化
sess.run(initialize_op)


# In[ ]:


get_ipython().system('tensorboard --logdir=/tmpc')


# In[ ]:




