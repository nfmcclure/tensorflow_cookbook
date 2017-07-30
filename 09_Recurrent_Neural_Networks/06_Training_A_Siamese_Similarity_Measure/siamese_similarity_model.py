# -*- coding: utf-8 -*-
# Siamese Address Similarity with TensorFlow (Model File)
#------------------------------------------
#
# Here, we show how to perform address matching
#   with a Siamese RNN model

import tensorflow as tf


def snn(address1, address2, dropout_keep_prob,
        vocab_size, num_features, input_length):
    
    # Define the siamese double RNN with a fully connected layer at the end
    def siamese_nn(input_vector, num_hidden):
        cell_unit = tf.contrib.rnn.BasicLSTMCell#tf.nn.rnn_cell.BasicLSTMCell
        
        # Forward direction cell
        lstm_forward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)
        
        # Backward direction cell
        lstm_backward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=dropout_keep_prob)
    
        # Split title into a character sequence
        input_embed_split = tf.split(axis=1, num_or_size_splits=input_length, value=input_vector)
        input_embed_split = [tf.squeeze(x, axis=[1]) for x in input_embed_split]
        
        # Create bidirectional layer
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                                    lstm_backward_cell,
                                                                    input_embed_split,
                                                                    dtype=tf.float32)
        except Exception:
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                              lstm_backward_cell,
                                                              input_embed_split,
                                                              dtype=tf.float32)
        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs) / input_length
        
        # Fully connected layer
        output_size = 10
        A = tf.get_variable(name="A", shape=[2*num_hidden, output_size],
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        
        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, dropout_keep_prob)
        
        return(final_output)
        
    output1 = siamese_nn(address1, num_features)
    # Declare that we will use the same variables on the second string
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        output2 = siamese_nn(address2, num_features)
    
    # Unit normalize the outputs
    output1 = tf.nn.l2_normalize(output1, 1)
    output2 = tf.nn.l2_normalize(output2, 1)
    # Return cosine distance
    #   in this case, the dot product of the norms is the same.
    dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)
    
    return(dot_prod)


def get_predictions(scores):
    predictions = tf.sign(scores, name="predictions")
    return(predictions)


def loss(scores, y_target, margin):
    # Calculate the positive losses
    pos_loss_term = 0.25 * tf.square(tf.subtract(1., scores))
    
    # If y-target is -1 to 1, then do the following
    pos_mult = tf.add(tf.multiply(0.5, tf.cast(y_target, tf.float32)), 0.5)
    # Else if y-target is 0 to 1, then do the following
    pos_mult = tf.cast(y_target, tf.float32)
    
    # Make sure positive losses are on similar strings
    positive_loss = tf.multiply(pos_mult, pos_loss_term)
    
    # Calculate negative losses, then make sure on dissimilar strings
    
    # If y-target is -1 to 1, then do the following:
    neg_mult = tf.add(tf.multiply(-0.5, tf.cast(y_target, tf.float32)), 0.5)
    # Else if y-target is 0 to 1, then do the following
    neg_mult = tf.subtract(1., tf.cast(y_target, tf.float32))
    
    negative_loss = neg_mult*tf.square(scores)
    
    # Combine similar and dissimilar losses
    loss = tf.add(positive_loss, negative_loss)
    
    # Create the margin term.  This is when the targets are 0.,
    #  and the scores are less than m, return 0.
    
    # Check if target is zero (dissimilar strings)
    target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)
    # Check if cosine outputs is smaller than margin
    less_than_margin = tf.less(scores, margin)
    # Check if both are true
    both_logical = tf.logical_and(target_zero, less_than_margin)
    both_logical = tf.cast(both_logical, tf.float32)
    # If both are true, then multiply by (1-1)=0.
    multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
    total_loss = tf.multiply(loss, multiplicative_factor)
    
    # Average loss over batch
    avg_loss = tf.reduce_mean(total_loss)
    return(avg_loss)


def accuracy(scores, y_target):
    predictions = get_predictions(scores)
    # Cast into integers (outputs can only be -1 or +1)
    y_target_int = tf.cast(y_target, tf.int32)
    # Change targets from (0,1) --> (-1, 1)
    #    via (2 * x - 1)
    #y_target_int = tf.sub(tf.mul(y_target_int, 2), 1)
    predictions_int = tf.cast(tf.sign(predictions), tf.int32)
    correct_predictions = tf.equal(predictions_int, y_target_int)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return(accuracy)