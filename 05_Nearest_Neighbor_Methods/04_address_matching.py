# Address Matching with k-Nearest Neighbors
#----------------------------------
#
# This function illustrates a way to perform
# address matching between two data sets.
#
# For each test address, we will return the
# closest reference address to it.
#
# We will consider two distance functions:
# 1) Edit distance for street number/name and
# 2) Euclidian distance (L2) for the zip codes

import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# First we generate the data sets we will need
# n = Size of created data sets
n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']
rand_zips = [random.randint(65000,65999) for i in range(5)]

# Function to randomly create one typo in a string w/ a probability
def create_typo(s, prob=0.75):
    if random.uniform(0,1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind]=random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return(s)

# Generate the reference dataset
numbers = [random.randint(1, 9999) for i in range(n)]
streets = [random.choice(street_names) for i in range(n)]
street_suffs = [random.choice(street_types) for i in range(n)]
zips = [random.choice(rand_zips) for i in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets,zips)]

# Generate test dataset with some typos
typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets,zips)]

# Now we can perform address matching
# Create graph
sess = tf.Session()

# Placeholders
test_address = tf.sparse_placeholder( dtype=tf.string)
test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)
ref_address = tf.sparse_placeholder(dtype=tf.string)
ref_zip = tf.placeholder(shape=[None, n], dtype=tf.float32)

# Declare Zip code distance for a test zip and reference set
zip_dist = tf.square(tf.sub(ref_zip, test_zip))

# Declare Edit distance for address
address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

# Create similarity scores
zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
zip_sim = tf.div(tf.sub(zip_max, zip_dist), tf.sub(zip_max, zip_min))
address_sim = tf.sub(1., address_dist)

# Combine distance functions
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.add(tf.transpose(tf.mul(address_weight, address_sim)), tf.mul(zip_weight, zip_sim))

# Predict: Get max similarity entry
top_match_index = tf.argmax(weighted_sim, 1)


# Function to Create a character-sparse tensor from strings
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_vec) for yi,y in enumerate(x)]
    chars = list(''.join(word_vec))
    return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

# Loop through test indices
reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# Create sparse address reference set
sparse_ref_set = sparse_from_word_vec(reference_addresses)

for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]
    
    # Create sparse address vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)
    
    feeddict={test_address: sparse_test_set,
               test_zip: test_zip_entry,
               ref_address: sparse_ref_set,
               ref_zip: reference_zips}
    best_match = sess.run(top_match_index, feed_dict=feeddict)
    best_street = reference_addresses[best_match[0]]
    [best_zip] = reference_zips[0][best_match]
    [[test_zip_]] = test_zip_entry
    print('Address: ' + str(test_address_entry) + ', ' + str(test_zip_))
    print('Match  : ' + str(best_street) + ', ' + str(best_zip))