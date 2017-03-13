# -*- coding: utf-8 -*-
#
# Creating Sequence to Sequence Model Class
#-------------------------------------
#  Here we implement the Seq2Seq class for modeling language translation
#

import numpy as np
import tensorflow as tf


# Declare Seq2seq translation model
class Seq2Seq(object):
    def __init__(self, vocab_size, x_buckets, y_buckets, rnn_size,
                 num_layers, max_gradient, batch_size, learning_rate,
                 lr_decay_rate, forward_only=False):
        self.vocab_size = vocab_size
        self.x_buckets = x_buckets
        self.y_buckets = y_buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay = self.learning_rate.assign(self.learning_rate * lr_decay_rate)
        self.global_step = tf.Variable(0, trainable=False)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        
        # Decoding function
        def decode_wrapper(x, y, forward_only):
            decode_fun = tf.nn.seq2seq.embedding_attention_seq2seq(x,
                y,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=rnn_size,
                feed_previous=forward_only,
                dtype=tf.float32)
            return(decode_fun)
        
        # Loss function
        loss_fun = tf.nn.softmax_cross_entropy_with_logits
        
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(x_buckets[-1]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{}".format(i)))
                                                      
        for i in range(x_buckets[-1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{}".format(i)))
        
        
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        xy_buckets = [x for x in zip(x_buckets, y_buckets)]
        
        if forward_only:
            self.outputs, self.loss =  tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, xy_buckets, lambda x, y: decode_wrapper(x, y, True),
                softmax_loss_function=loss_fun)
        else:
            self.outputs, self.loss =  tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, xy_buckets, lambda x, y: decode_wrapper(x, y, False),
                softmax_loss_function=loss_fun)
        
        # Gradients and SGD update operation for training the model.
        if not forward_only:
            # Initialize gradient and update functions
            self.gradient_norms = []
            self.update_funs = []
            # Create a optimizer to use for each data bucket
            my_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(xy_buckets)):
                # For each bucket, get the gradients
                gradients = tf.gradients(self.losses[b], tf.trainable_variables())
                # Clip the gradients
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                self.gradient_norms.append(norm)
                # Get the gradient update step for each variable
                temp_optimizer = my_optimizer.apply_gradients(zip(clipped_gradients, tf.trainable_variables()),
                                                              global_step=self.global_step)
                self.update_funs.append(temp_optimizer)

        self.saver = tf.train.Saver(tf.global_variables())
        
    # Define how to step forward (or backward) in the model
    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        for l in range(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
    
    # Define batch iteration function
    def batch_iter(self, data, bucket_num):
        decoder_len = self.y_buckets[bucket_num]
        encoder_len = self.x_buckets[bucket_num]
        batch_ix = np.random.choice(range(len(data[bucket_num])),
                                    size = self.batch_size)
        batch_data = [data[bucket_num][ix] for ix in batch_ix]
        encoder_inputs = [x[0] for x in batch_data]
        decoder_inputs = [x[1] for x in batch_data]
        
        # Pad encoder inputs with zeros
        encoder_inputs = [(x + [0]*encoder_len)[:encoder_len] for x in encoder_inputs]
        
        # Put a '1' at the start of decoder, and pad end with zeros
        decoder_inputs = [([1] + x + [0]*decoder_len)[:(decoder_len)] for x in decoder_inputs]
        
        # Transpose the inputs/outputs into list of arrays, each array is the i-th element
        encoder_inputs_t = [np.array(x) for x in zip(*encoder_inputs)]
        decoder_inputs_t = [np.array(x) for x in zip(*decoder_inputs)]
        
        # Create batch weights (0 for padding, 1 otherwise)
        target_weights = np.ones(shape=np.array(decoder_inputs_t).shape)
        zero_ix = [[(row, col) for col, c_val in enumerate(rows) if c_val==0] for row, rows in enumerate(decoder_inputs_t)]
        zero_ix = [val for sublist in zero_ix for val in sublist if sublist]
        # Set batch weights to zero
        for row, col in zero_ix:
            target_weights[row,col]=0
        
        # Need to roll the target weights (weights point to the next case)
        batch_weights = np.roll(batch_weights, -1, axis=0)
        # Last row should be all zeros
        target_weights[-1,:] = [0]*self.batch_size
        # Make weights a list of arrays
        target_weights = [np.array(r, dtype=np.float32) for r in target_weights]
        
        return(encoder_inputs_t, decoder_inputs_t, target_weights)