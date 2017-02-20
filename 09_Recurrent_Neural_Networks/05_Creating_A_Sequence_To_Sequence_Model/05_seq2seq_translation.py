# -*- coding: utf-8 -*-
#
# Creating Sequence to Sequence Models
#-------------------------------------
#  Here we show how to implement sequence to sequence models.
#  Specifically, we will build an English to German translation model.
#

import os
import re
import string
import requests
import io
import numpy as np
import collections
import random
import pickle
import string
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from collections import Counter
from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a session
sess = tf.Session()

# Model Parameters
learning_rate = 0.1
lr_decay_rate = 0.99
lr_decay_every = 100
max_gradient = 5.0
batch_size = 50
num_layers = 3
rnn_size = 500
layer_size = 512
generations = 10000
vocab_size = 10000
save_every = 1000
eval_every = 500
output_every = 50
punct = string.punctuation

# Data Parameters
data_dir = 'temp'
data_file = 'eng_ger.txt'
model_path = 'seq2seq_model'
full_model_dir = os.path.join(data_dir, model_path)

# Test Translation from English (lowercase, no punct)
test_english = ['hello where is my computer',
                'the quick brown fox jumped over the lazy dog',
                'is it going to rain tomorrow']

# Make Model Directory
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)

# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print('Loading English-German Data')
# Check for data, if it doesn't exist, download it and save it
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Data not found, downloading Eng-Ger sentences from www.manythings.org')
    sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
    r = requests.get(sentence_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('deu.txt')
    # Format Data
    eng_ger_data = file.decode()
    eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
    eng_ger_data = eng_ger_data.decode().split('\n')
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        for sentence in eng_ger_data:
            out_conn.write(sentence + '\n')
else:
    eng_ger_data = []
    with open(os.path.join(data_dir, data_file), 'r') as in_conn:
        for row in in_conn:
            eng_ger_data.append(row[:-1])

# Remove punctuation
eng_ger_data = [''.join(char for char in sent if char not in punct) for sent in eng_ger_data]
# Split each sentence by tabs    
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
english_sentence = [x.lower().split() for x in english_sentence]
german_sentence = [x.lower().split() for x in german_sentence]

print('Processing the vocabularies.')
# Process the English Vocabulary
all_english_words = [word for sentence in english_sentence for word in sentence]
all_english_counts = Counter(all_english_words)
eng_word_keys = [x[0] for x in all_english_counts.most_common(vocab_size-1)] #-1 because 0=unknown is also in there
eng_vocab2ix = dict(zip(eng_word_keys, range(1,vocab_size)))
eng_ix2vocab = {val:key for key, val in eng_vocab2ix.items()}
english_processed = []
for sent in english_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except:
            temp_sentence.append(0)
    english_processed.append(temp_sentence)


# Process the German Vocabulary
all_german_words = [word for sentence in german_sentence for word in sentence]
all_german_counts = Counter(all_german_words)
ger_word_keys = [x[0] for x in all_german_counts.most_common(vocab_size-1)]
ger_vocab2ix = dict(zip(ger_word_keys, range(1,vocab_size)))
ger_ix2vocab = {val:key for key, val in ger_vocab2ix.items()}
german_processed = []
for sent in german_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(ger_vocab2ix[word])
        except:
            temp_sentence.append(0)
    german_processed.append(temp_sentence)


# Process the test english sentences, use '0' if word not in our vocab
test_data = []
for sentence in test_english:
    temp_sentence = []
    for word in sentence.split(' '):
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except:
            # Use '0' if the word isn't in our vocabulary
            temp_sentence.append(0)
    test_data.append(temp_sentence)

# Define Buckets for sequence lengths
# We will split data into the corresponding buckets:
# (x1, y1), (x2, y2), ...
# Where all entries in bucket 1: len(x)<x1 and len(y)<y1 and so on.
x_maxs = [5, 7, 11, 50]
y_maxs = [10, 12, 17, 60]
buckets = [x for x in zip(x_maxs, y_maxs)]
bucketed_data = [[] for _ in range(len(x_maxs))]
for eng, ger in zip(english_processed, german_processed):
    for ix, (x_max, y_max) in enumerate(zip(x_maxs, y_maxs)):
        if (len(eng) <= x_max) and (len(ger) <= y_max):
            bucketed_data[ix].append([eng, ger])
            break

# Print summaries of buckets
train_bucket_sizes = [len(bucketed_data[b]) for b in range(len(buckets))]
train_total_size = float(sum(train_bucket_sizes))
for ix, bucket in enumerate(bucketed_data):
    print('Data pts in bucket {}: {}'.format(ix, len(bucket)))


# Create sequence to sequence model
def translation_model(sess, input_vocab_size, output_vocab_size,
                      buckets, rnn_size, num_layers, max_gradient,
                      learning_rate, lr_decay_rate, forward_only):
    model = seq2seq_model.Seq2SeqModel(
          input_vocab_size,
          output_vocab_size,
          buckets,
          rnn_size,
          num_layers,
          max_gradient,
          batch_size,
          learning_rate,
          lr_decay_rate,
          forward_only=forward_only,
          dtype=tf.float32)
    return(model)

print('Creating Translation Model')
input_vocab_size = vocab_size
output_vocab_size = vocab_size
with tf.variable_scope('translate_model') as scope:
    translate_model = translation_model(sess, vocab_size, vocab_size,
                                        buckets, rnn_size, num_layers,
                                        max_gradient, learning_rate,
                                        lr_decay_rate, False)
    #Reuse the variables for the test model
    scope.reuse_variables()
    test_model = translation_model(sess, vocab_size, vocab_size,
                                   buckets, rnn_size, num_layers,
                                   max_gradient, learning_rate,
                                   lr_decay_rate, True)
    test_model.batch_size = 1

# Initialize all model variables
init = tf.global_variables_initializer()
sess.run(init)

# Start training
train_loss = []
for i in range(generations):
    rand_bucket_ix = np.random.choice(len(bucketed_data))
    
    model_outputs = translate_model.get_batch(bucketed_data, rand_bucket_ix)
    encoder_inputs, decoder_inputs, target_weights = model_outputs
    
    # Get the (gradient norm, loss, and outputs)
    _, step_loss, _ = translate_model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, rand_bucket_ix, False)
    
    # Output status
    if (i+1) % output_every == 0:
        train_loss.append(step_loss)
        print('Gen #{} out of {}. Loss: {:.4}'.format(i+1, generations, step_loss))
    
    # Check if we should decay the learning rate
    if (i+1) % lr_decay_every == 0:
        sess.run(translate_model.learning_rate_decay_op)
        
    # Save model
    if (i+1) % save_every == 0:
        print('Saving model to {}.'.format(full_model_dir))
        model_save_path = os.path.join(full_model_dir, "eng_ger_translation.ckpt")
        translate_model.saver.save(sess, model_save_path, global_step=i)
        
    # Eval on test set
    if (i+1) % eval_every == 0:
        for ix, sentence in enumerate(test_data):
            # Find which bucket sentence goes in
            bucket_id = next(index for index, val in enumerate(x_maxs) if val>=len(sentence))
            # Get RNN model outputs
            encoder_inputs, decoder_inputs, target_weights = test_model.get_batch(
                {bucket_id: [(sentence, [])]}, bucket_id)
            # Get logits
            _, test_loss, output_logits = test_model.step(sess, encoder_inputs, decoder_inputs,
                                                           target_weights, bucket_id, True)
            ix_output = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is a 0 symbol in outputs end the output there.
            ix_output = ix_output[0:[ix for ix, x in enumerate(ix_output+[0]) if x==0][0]]
            # Get german words from indices
            test_german = [ger_ix2vocab[x] for x in ix_output]
            print('English: {}'.format(test_english[ix]))
            print('German: {}'.format(test_german))
            

# Plot train loss
loss_generations = [i for i in range(generations) if i%output_every==0]
plt.plot(loss_generations, train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()