# -*- coding: utf-8 -*-
# Tensorflow Production Example (Evaluating)
#----------------------------------
#
# We pull together everything and create an example
#    of best tensorflow production tips
#
# The example we will productionalize is the spam/ham RNN
#    from the RNN Chapter.

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

tf.app.flags.DEFINE_string("storage_folder", "temp", "Where to store model and data.")
tf.app.flags.DEFINE_string('model_file', False, 'Model file location.')
tf.app.flags.DEFINE_boolean('run_unit_tests', False, 'If true, run tests.')
FLAGS = tf.app.flags.FLAGS


# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)


# Load vocab processor
def load_vocab():
    vocab_path = os.path.join(FLAGS.storage_folder, "vocab")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    return(vocab_processor)


# Process input data:
def process_data(input_data, vocab_processor):
    input_data = clean_text(input_data)
    input_data = input_data.split()
    processed_input = np.array(list(vocab_processor.transform(input_data)))
    return(processed_input)


# Get input function
def get_input_data():
    """
    For this function, we just prompt the user for a text message to evaluate
        But this function could also potentially read a file in as well.
    """
    input_text = input("Please enter a text message to evaluate: ")
    vocab_processor = load_vocab()
    return(process_data(input_text, vocab_processor))


# Test clean_text function
class clean_test(tf.test.TestCase):
    # Make sure cleaning function behaves correctly
    def clean_string_test(self):
        with self.test_session():
            test_input = '--Tensorflow\'s so Great! Don\t you think so?   '
            test_expected = 'tensorflows so great don you think so'
            test_out = clean_text(test_input)
            self.assertEqual(test_expected, test_out)


# Main function
def main(args):
    # Get flags
    storage_folder = FLAGS.storage_folder
    
    # Get user input text
    x_data = get_input_data()
    
    # Load model
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(os.path.join(storage_folder, "model.ckpt")))
            saver.restore(sess, os.path.join(storage_folder, "model.ckpt"))

            # Get the placeholders from the graph by name
            x_data_ph = graph.get_operation_by_name("x_data_ph").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            probability_outputs = graph.get_operation_by_name("probability_outputs").outputs[0]

            # Make the prediction
            eval_feed_dict = {x_data_ph: x_data, dropout_keep_prob: 1.0}
            probability_prediction = sess.run(tf.reduce_mean(probability_outputs, 0), eval_feed_dict)
            
            # Print output (Or save to file or DB connection?)
            print('Probability of Spam: {:.4}'.format(probability_prediction[1]))

# Run main module/tf App
if __name__ == "__main__":
    if FLAGS.run_unit_tests:
        # Perform unit tests
        tf.test.main()
    else:
        # Run evaluation
        tf.app.run()