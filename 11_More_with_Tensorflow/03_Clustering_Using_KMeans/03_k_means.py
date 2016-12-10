# -*- coding: utf-8 -*-
# K-means with Tensorflow
#----------------------------------
#
# This script shows how to do k-means with Tensorflow

import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

iris = datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.data[0])
print(set(iris.target))

num_pts = len(iris.data)
num_feats = len(iris.data[0])

# Set k-means parameters
# There are 3 types of iris flowers, see if we can predict them
k=3 
generations = 25

data_points = tf.Variable(iris.data)
cluster_labels = tf.Variable(tf.zeros([num_pts], dtype=tf.int64))

# Randomly choose starting points
rand_starts = np.array([iris.data[np.random.choice(len(iris.data))] for _ in range(k)])

centroids = tf.Variable(rand_starts)

# In order to calculate the distance between every data point and every centroid, we
#  repeat the centroids into a (num_points) by k matrix.
centroid_matrix = tf.reshape(tf.tile(centroids, [num_pts, 1]), [num_pts, k, num_feats])
# Then we reshape the data points into k (3) repeats
point_matrix = tf.reshape(tf.tile(data_points, [1, k]), [num_pts, k, num_feats])
distances = tf.reduce_sum(tf.square(point_matrix - centroid_matrix), reduction_indices=2)

#Find the group it belongs to with tf.argmin()
centroid_group = tf.argmin(distances, 1)

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return(total / count)

means = bucket_mean(data_points, centroid_group, k)

do_updates = tf.group(centroids.assign(means), cluster_labels.assign(centroid_group))

init = tf.initialize_all_variables()

sess.run(init)

for i in range(generations):
    print('Calculating gen {}, out of {}.'.format(i, generations))
    _, centroid_group_count = sess.run([do_updates, centroid_group])
    group_count = []
    for ix in range(k):
        group_count.append(np.sum(centroid_group_count==ix))
    print('Group counts: {}'.format(group_count))
    

[centers, assignments] = sess.run([centroids, cluster_labels])

# Find which group assignments correspond to which group labels
# First, need a most common element function
def most_common(my_list):
    return(max(set(my_list), key=my_list.count))

label0 = most_common(list(assignments[0:50]))
label1 = most_common(list(assignments[50:100]))
label2 = most_common(list(assignments[100:150]))

group0_count = np.sum(assignments[0:50]==label0)
group1_count = np.sum(assignments[50:100]==label1)
group2_count = np.sum(assignments[100:150]==label2)

accuracy = (group0_count + group1_count + group2_count)/150.

print('Accuracy: {:.2}'.format(accuracy))