# -*- coding: utf-8 -*-
#
# Creating Sequence to Sequence Models
#-------------------------------------
#  Here we show how to implement sequence to sequence models.
#  Specifically, we will build an English to German translation model.
#

import os
import re
import sys
import json
import math
import time
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
from tensorflow.python.ops import lookup_ops
from tensorflow.python.framework import ops
ops.reset_default_graph()

local_repository = 'temp/seq2seq'

# models can be retrieved from github: https://github.com/tensorflow/models.git
# put the models dir under python search lib path.

if not os.path.exists(local_repository):
    from git import Repo
    tf_model_repository = 'https://github.com/tensorflow/nmt/'
    Repo.clone_from(tf_model_repository, local_repository)
    sys.path.insert(0, 'temp/seq2seq/nmt/')

# May also try to use 'attention model' by importing the attention model:
# from temp.seq2seq.nmt import attention_model as attention_model
from temp.seq2seq.nmt import model as model
from temp.seq2seq.nmt.utils import vocab_utils as vocab_utils
import temp.seq2seq.nmt.model_helper as model_helper
import temp.seq2seq.nmt.utils.iterator_utils as iterator_utils
import temp.seq2seq.nmt.utils.misc_utils as utils
import temp.seq2seq.nmt.train as train

# Start a session
sess = tf.Session()

# Model Parameters
vocab_size = 10000
punct = string.punctuation

# Data Parameters
data_dir = 'temp'
data_file = 'eng_ger.txt'
model_path = 'seq2seq_model'
full_model_dir = os.path.join(data_dir, model_path)

# Load hyper-parameters for translation model. (Good defaults are provided in Repository).
hparams = tf.contrib.training.HParams()
param_file = 'temp/seq2seq/nmt/standard_hparams/wmt16.json'
# Can also try: (For different architectures)
# 'temp/seq2seq/nmt/standard_hparams/iwslt15.json'
# 'temp/seq2seq/nmt/standard_hparams/wmt16_gnmt_4_layer.json',
# 'temp/seq2seq/nmt/standard_hparams/wmt16_gnmt_8_layer.json',

with open(param_file, "r") as f:
    params_json = json.loads(f.read())

for key, value in params_json.items():
    hparams.add_hparam(key, value)
hparams.add_hparam('num_gpus', 0)
hparams.add_hparam('num_encoder_layers', hparams.num_layers)
hparams.add_hparam('num_decoder_layers', hparams.num_layers)
hparams.add_hparam('num_encoder_residual_layers', 0)
hparams.add_hparam('num_decoder_residual_layers', 0)
hparams.add_hparam('init_op', 'uniform')
hparams.add_hparam('random_seed', None)
hparams.add_hparam('num_embeddings_partitions', 0)
hparams.add_hparam('warmup_steps', 0)
hparams.add_hparam('length_penalty_weight', 0)
hparams.add_hparam('sampling_temperature', 0.0)
hparams.add_hparam('num_translations_per_input', 1)
hparams.add_hparam('warmup_scheme', 't2t')
hparams.add_hparam('epoch_step', 0)
hparams.num_train_steps = 5000

# Not use any pretrained embeddings
hparams.add_hparam('src_embed_file', '')
hparams.add_hparam('tgt_embed_file', '')
hparams.add_hparam('num_keep_ckpts', 5)
hparams.add_hparam('avg_ckpts', False)

# Remove attention
hparams.attention = None

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
    eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
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
print('Done!')

# Remove punctuation
eng_ger_data = [''.join(char for char in sent if char not in punct) for sent in eng_ger_data]
# Split each sentence by tabs    
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
english_sentence = [x.lower().split() for x in english_sentence]
german_sentence = [x.lower().split() for x in german_sentence]

# We need to write them to separate text files for the text-line-dataset operations.
train_prefix = 'train'
src_suffix = 'en'  # English
tgt_suffix = 'de'  # Deutsch (German)
source_txt_file = train_prefix + '.' + src_suffix
hparams.add_hparam('src_file', source_txt_file)
target_txt_file = train_prefix + '.' + tgt_suffix
hparams.add_hparam('tgt_file', target_txt_file)
with open(source_txt_file, 'w') as f:
    for sent in english_sentence:
        f.write(' '.join(sent) + '\n')

with open(target_txt_file, 'w') as f:
    for sent in german_sentence:
        f.write(' '.join(sent) + '\n')


# Partition some sentences off for testing files
test_prefix = 'test_sent'
hparams.add_hparam('dev_prefix', test_prefix)
hparams.add_hparam('train_prefix', train_prefix)
hparams.add_hparam('test_prefix', test_prefix)
hparams.add_hparam('src', src_suffix)
hparams.add_hparam('tgt', tgt_suffix)

num_sample = 100
total_samples = len(english_sentence)
# Get around 'num_sample's every so often in the src/tgt sentences
ix_sample = [x for x in range(total_samples) if x % (total_samples // num_sample) == 0]
test_src = [' '.join(english_sentence[x]) for x in ix_sample]
test_tgt = [' '.join(german_sentence[x]) for x in ix_sample]

# Write test sentences to file
with open(test_prefix + '.' + src_suffix, 'w') as f:
    for eng_test in test_src:
        f.write(eng_test + '\n')

with open(test_prefix + '.' + tgt_suffix, 'w') as f:
    for ger_test in test_src:
        f.write(ger_test + '\n')

print('Processing the vocabularies.')
# Process the English Vocabulary
all_english_words = [word for sentence in english_sentence for word in sentence]
all_english_counts = Counter(all_english_words)
eng_word_keys = [x[0] for x in all_english_counts.most_common(vocab_size-3)]  # -3 because UNK, S, /S is also in there
eng_vocab2ix = dict(zip(eng_word_keys, range(1, vocab_size)))
eng_ix2vocab = {val: key for key, val in eng_vocab2ix.items()}
english_processed = []
for sent in english_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except KeyError:
            temp_sentence.append(0)
    english_processed.append(temp_sentence)


# Process the German Vocabulary
all_german_words = [word for sentence in german_sentence for word in sentence]
all_german_counts = Counter(all_german_words)
ger_word_keys = [x[0] for x in all_german_counts.most_common(vocab_size-3)]  # -3 because UNK, S, /S is also in there
ger_vocab2ix = dict(zip(ger_word_keys, range(1, vocab_size)))
ger_ix2vocab = {val: key for key, val in ger_vocab2ix.items()}
german_processed = []
for sent in german_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(ger_vocab2ix[word])
        except KeyError:
            temp_sentence.append(0)
    german_processed.append(temp_sentence)


# Save vocab files for data processing
source_vocab_file = 'vocab' + '.' + src_suffix
hparams.add_hparam('src_vocab_file', source_vocab_file)
eng_word_keys = ['<unk>', '<s>', '</s>'] + eng_word_keys

target_vocab_file = 'vocab' + '.' + tgt_suffix
hparams.add_hparam('tgt_vocab_file', target_vocab_file)
ger_word_keys = ['<unk>', '<s>', '</s>'] + ger_word_keys

# Write out all unique english words
with open(source_vocab_file, 'w') as f:
    for eng_word in eng_word_keys:
        f.write(eng_word + '\n')

# Write out all unique german words
with open(target_vocab_file, 'w') as f:
    for ger_word in ger_word_keys:
        f.write(ger_word + '\n')

# Add vocab size to hyper parameters
hparams.add_hparam('src_vocab_size', vocab_size)
hparams.add_hparam('tgt_vocab_size', vocab_size)

# Add out-directory
out_dir = 'temp/seq2seq/nmt_out'
hparams.add_hparam('out_dir', out_dir)
if not tf.gfile.Exists(out_dir):
    tf.gfile.MakeDirs(out_dir)


class TrainGraph(collections.namedtuple("TrainGraph", ("graph", "model", "iterator", "skip_count_placeholder"))):
    pass


def create_train_graph(scope=None):
    graph = tf.Graph()
    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(hparams.src_vocab_file,
                                                                           hparams.tgt_vocab_file,
                                                                           share_vocab=False)

        src_dataset = tf.data.TextLineDataset(hparams.src_file)
        tgt_dataset = tf.data.TextLineDataset(hparams.tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        iterator = iterator_utils.get_iterator(src_dataset, tgt_dataset, src_vocab_table, tgt_vocab_table,
                                               batch_size=hparams.batch_size,
                                               sos=hparams.sos,
                                               eos=hparams.eos,
                                               random_seed=None,
                                               num_buckets=hparams.num_buckets,
                                               src_max_len=hparams.src_max_len,
                                               tgt_max_len=hparams.tgt_max_len,
                                               skip_count=skip_count_placeholder)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.TRAIN,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  scope=scope)

    return TrainGraph(graph=graph, model=final_model, iterator=iterator, skip_count_placeholder=skip_count_placeholder)


train_graph = create_train_graph()


# Create the evaluation graph
class EvalGraph(collections.namedtuple("EvalGraph", ("graph", "model", "src_file_placeholder", "tgt_file_placeholder",
                                                     "iterator"))):
    pass


def create_eval_graph(scope=None):
    graph = tf.Graph()

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
            hparams.src_vocab_file, hparams.tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.EVAL,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  scope=scope)
    return EvalGraph(graph=graph,
                     model=final_model,
                     src_file_placeholder=src_file_placeholder,
                     tgt_file_placeholder=tgt_file_placeholder,
                     iterator=iterator)


eval_graph = create_eval_graph()


# Inference graph
class InferGraph(
    collections.namedtuple("InferGraph", ("graph", "model", "src_placeholder", "batch_size_placeholder", "iterator"))):
    pass


def create_infer_graph(scope=None):
    graph = tf.Graph()
    with graph.as_default():
        src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(hparams.src_vocab_file,
                                                                           hparams.tgt_vocab_file,
                                                                           hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(hparams.tgt_vocab_file,
                                                                             default_value=vocab_utils.UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)
        iterator = iterator_utils.get_infer_iterator(src_dataset,
                                                     src_vocab_table,
                                                     batch_size=batch_size_placeholder,
                                                     eos=hparams.eos,
                                                     src_max_len=hparams.src_max_len_infer)
        final_model = model.Model(hparams,
                                  iterator=iterator,
                                  mode=tf.contrib.learn.ModeKeys.INFER,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  reverse_target_vocab_table=reverse_tgt_vocab_table,
                                  scope=scope)
    return InferGraph(graph=graph,
                      model=final_model,
                      src_placeholder=src_placeholder,
                      batch_size_placeholder=batch_size_placeholder,
                      iterator=iterator)


infer_graph = create_infer_graph()


# Create sample data for evaluation
sample_ix = [25, 125, 240, 450]
sample_src_data = [' '.join(english_sentence[x]) for x in sample_ix]
sample_tgt_data = [' '.join(german_sentence[x]) for x in sample_ix]

config_proto = utils.get_config_proto()

train_sess = tf.Session(config=config_proto, graph=train_graph.graph)
eval_sess = tf.Session(config=config_proto, graph=eval_graph.graph)
infer_sess = tf.Session(config=config_proto, graph=infer_graph.graph)

# Load the training graph
with train_graph.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(train_graph.model,
                                                                        hparams.out_dir,
                                                                        train_sess,
                                                                        "train")


summary_writer = tf.summary.FileWriter(os.path.join(hparams.out_dir, 'Training'), train_graph.graph)

for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)


eval_output = train.run_full_eval(hparams.out_dir, infer_graph, infer_sess, eval_graph, eval_sess,
                                  hparams, summary_writer, sample_src_data, sample_tgt_data)

eval_results, _, acc_blue_scores = eval_output

# Training Initialization
last_stats_step = global_step
last_eval_step = global_step
last_external_eval_step = global_step

steps_per_eval = 10 * hparams.steps_per_stats
steps_per_external_eval = 5 * steps_per_eval

avg_step_time = 0.0
step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
checkpoint_total_count = 0.0
speed, train_ppl = 0.0, 0.0

utils.print_out("# Start step %d, lr %g, %s" %
                (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
                 time.ctime()))
skip_count = hparams.batch_size * hparams.epoch_step
utils.print_out("# Init train iterator, skipping %d elements" % skip_count)

train_sess.run(train_graph.iterator.initializer,
              feed_dict={train_graph.skip_count_placeholder: skip_count})


# Run training
while global_step < hparams.num_train_steps:
    start_time = time.time()
    try:
        step_result = loaded_train_model.train(train_sess)
        (_, step_loss, step_predict_count, step_summary, global_step, step_word_count,
         batch_size, __, ___) = step_result
        hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
        # Next Epoch
        hparams.epoch_step = 0
        utils.print_out("# Finished an epoch, step %d. Perform external evaluation" % global_step)
        train.run_sample_decode(infer_graph,
                                infer_sess,
                                hparams.out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_scores, test_scores, _ = train.run_external_eval(infer_graph,
                                                             infer_sess,
                                                             hparams.out_dir,
                                                             hparams,
                                                             summary_writer)
        train_sess.run(train_graph.iterator.initializer, feed_dict={train_graph.skip_count_placeholder: 0})
        continue

    summary_writer.add_summary(step_summary, global_step)

    # Statistics
    step_time += (time.time() - start_time)
    checkpoint_loss += (step_loss * batch_size)
    checkpoint_predict_count += step_predict_count
    checkpoint_total_count += float(step_word_count)

    # print statistics
    if global_step - last_stats_step >= hparams.steps_per_stats:
        last_stats_step = global_step
        avg_step_time = step_time / hparams.steps_per_stats
        train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
        speed = checkpoint_total_count / (1000 * step_time)

        utils.print_out("  global step %d lr %g "
                        "step-time %.2fs wps %.2fK ppl %.2f %s" %
                        (global_step,
                         loaded_train_model.learning_rate.eval(session=train_sess),
                         avg_step_time, speed, train_ppl, train._get_best_results(hparams)))

        if math.isnan(train_ppl):
            break

        # Reset timer and loss.
        step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
        checkpoint_total_count = 0.0

    if global_step - last_eval_step >= steps_per_eval:
        last_eval_step = global_step
        utils.print_out("# Save eval, global step %d" % global_step)
        utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

        # Save checkpoint
        loaded_train_model.saver.save(train_sess, os.path.join(hparams.out_dir, "translate.ckpt"),
                                      global_step=global_step)

        # Evaluate on dev/test
        train.run_sample_decode(infer_graph,
                                infer_sess,
                                out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_ppl, test_ppl = train.run_internal_eval(eval_graph,
                                                    eval_sess,
                                                    out_dir,
                                                    hparams,
                                                    summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
        last_external_eval_step = global_step

        # Save checkpoint
        loaded_train_model.saver.save(train_sess, os.path.join(hparams.out_dir, "translate.ckpt"),
                                      global_step=global_step)

        train.run_sample_decode(infer_graph,
                                infer_sess,
                                out_dir,
                                hparams,
                                summary_writer,
                                sample_src_data,
                                sample_tgt_data)
        dev_scores, test_scores, _ = train.run_external_eval(infer_graph,
                                                             infer_sess,
                                                             out_dir,
                                                             hparams,
                                                             summary_writer)