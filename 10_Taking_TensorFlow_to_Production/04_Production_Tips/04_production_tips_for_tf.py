# -*- coding: utf-8 -*-
# Tips for TensorFlow to Production
#----------------------------------
#
# Various Tips for Taking TensorFlow to Production

############################################
#
# THIS SCRIPT IS NOT RUNNABLE.
#  -it only contains tips for production code
#
############################################

# Also you can clear the default graph from memory
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Saving Models
# File types created from saving:    
# checkpoint file:  Holds info on where the most recent models are
# events file:      Strictly for viewing graph in Tensorboard
# pbtxt file:       Textual protobufs file (uncompressed), used for debugging
# chkp file:        Holds data and model weights (large file)
# meta chkp files:  Model Graph and Meta-data (learning rate and operations)


# Saving data pipeline structures (vocabulary, )
word_list = ['to', 'be', 'or', 'not', 'to', 'be']
vocab_list = list(set(word_list))
vocab2ix_dict = dict(zip(vocab_list, range(len(vocab_list))))
ix2vocab_dict = {val:key for key,val in vocab2ix_dict.items()}

# Save vocabulary
import json
with open('vocab2ix_dict.json', 'w') as file_conn:
    json.dump(vocab2ix_dict, file_conn)

# Load vocabulary
with open('vocab2ix_dict.json', 'r') as file_conn:
    vocab2ix_dict = json.load(file_conn)

# After model declaration, add a saving operations
saver = tf.train.Saver()
# Then during training, save every so often, referencing the training generation
for i in range(generations):
    ...
    if i%save_every == 0:
        saver.save(sess, 'my_model', global_step=step)

# Can also save only specific variables:
saver = tf.train.Saver({"my_var": my_variable})


# other options for saver are 'keep checkpoint_every_n_hours'
#      also 'max_to_keep'= default 5.
        
# Be sure to name operations, and variables for easy loading for referencing later
conv_weights = tf.Variable(tf.random_normal(), name='conv_weights')
loss = tf.reduce_mean(... , name='loss')

# Instead of tyring argparse and main(), TensorFlow provides an 'app' function
#  to handle running and loading of arguments

# At the beginning of the file, define the flags.
tf.flags.DEFINE_string("worker_locations", "", "List of worker addresses.")
tf.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.flags.DEFINE_integer('generations', 1000, 'Number of training generations.')
tf.flags.DEFINE_boolean('run_unit_tests', False, 'If true, run tests.')
FLAGS = tf.flags.FLAGS

# Need to define a 'main' function for the app to run
def main(_):
    worker_ips = FLAGS.worker_locations.split(",")
    learning_rate = FLAGS.learning_rate
    generations = FLAGS.generations
    run_unit_tests = FLAGS.run_unit_tests

# Run the TensorFlow app
if __name__ == "__main__":
    # The following is looking for a "main()" function to run and will pass.
    tf.app.run()
    # Can modify this to be more custom:
    tf.app.run(main=my_main_function(), argv=my_arguments)


# Use of TensorFlow's built in logging:
# Five levels: DEBUG, INFO, WARN, ERROR, and FATAL
tf.logging.set_verbosity(tf.logging.WARN)
# WARN is the default value, but to see more information, you can set it to
#    INFO or DEBUG
tf.logging.set_verbosity(tf.logging.DEBUG)
# Note: 'DEBUG' is quite verbose.

