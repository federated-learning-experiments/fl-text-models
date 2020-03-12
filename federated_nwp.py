# Copyright 2020, Joel Stremmel and Arjun Singh.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
**Word Level Federated Text Generation with Stack Overflow:**

- Last updated 03-09-20
- Runs on GCP and local Ubuntu 16.04

**About:**

This notebook loads the Stack Overflow data available through
`tff.simulation.datasets` and trains an LSTM model with
Federared Averaging by following the Federated Learning
for Text Generation example notebook at
https://github.com/tensorflow/federated/blob/master/docs/tutorials.

**Loading Pretrained Embeddings:**

The embedding layer is initialized with one of the following options
by setting the embedding_layer parameter:
- GloVe: https://nlp.stanford.edu/projects/glove/
    - license: https://www.opendatacommons.org/licenses/pddl/1.0/
- FastText: https://fasttext.cc/docs/en/english-vectors.html
    - license: https://creativecommons.org/licenses/by-sa/3.0/
- GPT-2: https://openai.com/blog/better-language-models/
    - license: https://github.com/huggingface/transformers/blob/master/LICENSE
- Random: https://tensorflow.org/api_docs/python/tf/random_uniform_initializer

After downloading the GloVe or FastText embeddings, place the embedding files
at the top level of the repository in directories called `word_embedding/glove`
and `word_embedding/fasttext` respectively.
GPT-2 embeddings are downloaded by running this file
which makes a call to `src/embeddings.py` to download the embeddings
from huggingface: https://github.com/huggingface/transformers.

**Environment Setup References:**

- Installing Tensorflow for GPU:
    https://www.tensorflow.org/install/gpu
- Install CUDA 10.0 and cuDNN v7.4.2 on Ubuntu 16.04:
    https://gist.github.com/matheustguimaraes/43e0b65aa534db4df2918f835b9b361d
- Tensorflow build configs:
    https://www.tensorflow.org/install/source#tested_build_configurations
"""

import nest_asyncio
nest_asyncio.apply()

import os, sys, io
sys.path.append(os.getcwd())

import json
import collections
import functools
import six
import time
import string
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_federated as tff

from src import (dataset, metrics, embeddings, model, validation,
    federated, generate_text, transfer_learning)

# Load Parameters
with open("params.json", "r") as read_file:
    params = json.load(read_file)

# Set Parameters
VOCAB_SIZE = params['VOCAB_SIZE']
BATCH_SIZE = params['BATCH_SIZE']
CLIENTS_EPOCHS_PER_ROUND = params['CLIENTS_EPOCHS_PER_ROUND']
MAX_SEQ_LENGTH = params['MAX_SEQ_LENGTH']
MAX_ELEMENTS_PER_USER = params['MAX_ELEMENTS_PER_USER']
CENTRALIZED_TRAIN = params['CENTRALIZED_TRAIN']
SHUFFLE_BUFFER_SIZE = params['SHUFFLE_BUFFER_SIZE']
NUM_VALIDATION_EXAMPLES = params['NUM_VALIDATION_EXAMPLES']
NUM_TEST_EXAMPLES = params['NUM_TEST_EXAMPLES']
NUM_PRETRAINING_ROUNDS = params['NUM_PRETRAINING_ROUNDS']
NUM_ROUNDS = params['NUM_ROUNDS']
NUM_TRAIN_CLIENTS = params['NUM_TRAIN_CLIENTS']
EMBEDDING_DIM = params['EMBEDDING_DIM']
RNN_UNITS = params['RNN_UNITS']
EMBEDDING_LAYER = params['EMBEDDING_LAYER']

# Set Save Path
sav = 'experiment_runs/{}_{}_{}_{}_{}/'.format(
    NUM_PRETRAINING_ROUNDS,
    EMBEDDING_LAYER,
    EMBEDDING_DIM,
    RNN_UNITS,
    EMBEDDING_DIM)

# Set Extended Vocab Size Using Special Tokens
extended_vocab_size = VOCAB_SIZE + len(
    dataset.get_special_tokens(VOCAB_SIZE))

# Create the Output Directory if Nonexistent
if not os.path.exists(sav):
    os.makedirs(sav)

# Load and Preprocess Word Level Datasets
train_data, val_data, test_data = dataset.construct_word_level_datasets(
    vocab_size=VOCAB_SIZE,
    batch_size=BATCH_SIZE,
    client_epochs_per_round=CLIENTS_EPOCHS_PER_ROUND,
    max_seq_len=MAX_SEQ_LENGTH,
    max_elements_per_user=MAX_ELEMENTS_PER_USER,
    centralized_train=CENTRALIZED_TRAIN,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    num_validation_examples=NUM_VALIDATION_EXAMPLES,
    num_test_examples=NUM_TEST_EXAMPLES)

# Retrieve the Fine Tunining Dataset Vocab
vocab = dataset.get_vocab(vocab_size=VOCAB_SIZE)

# Pretrain with a Different Text Corpus by First Reading in the Text Data
if NUM_PRETRAINING_ROUNDS > 0:

    # Load and preprocess the shakespeare dataset
    X, Y, vocab_sp = transfer_learning.load_and_preprocess_shakespeare(
        VOCAB_SIZE)

    # Build an embedding layer given the desired layer type
    embedding_matrix = embeddings.build_embedding_layer(
        embedding_type=EMBEDDING_LAYER,
        embedding_dim=EMBEDDING_DIM,
        vocab=vocab_sp)

    # Create the model to pretrain
    keras_model_sp = model.build_model(
        extended_vocab_size=extended_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        rnn_units=RNN_UNITS)

    # Compile the model
    evaluation_metrics_sp = validation.get_metrics(VOCAB_SIZE)
    model.compile_model(keras_model_sp, evaluation_metrics_sp)

    # Fit the model
    history = keras_model_sp.fit(X, Y, epochs=NUM_PRETRAINING_ROUNDS)

# Create Embedding Matrix for the Federated Model:
# If the model has been pretrained,
# this layer will be replaced during the transfer learning step.
embedding_matrix = embeddings.build_embedding_layer(
    embedding_type=EMBEDDING_LAYER,
    embedding_dim=EMBEDDING_DIM,
    vocab=vocab)

# Create TFF Version of the Model to be Trained with Federated Averaging:
# - TFF uses a sample batch to know the types and shapes the model expects.
# - The model function builds and compiles the model
#   and creates a TFF version to be trained.
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(val_data)))

# Initialize Train and Validation Model Trackers to be Used Below
evaluation_metric_names = ['loss',
                           'num_tokens',
                           'num_tokens_no_oov',
                           'num_batches',
                           'num_examples',
                           'accuracy',
                           'accuracy_no_oov',
                           'accuracy_no_oov_no_eos']

train_metrics_tracker = validation.model_history_tracker(
    metric_names=evaluation_metric_names)
val_metrics_tracker = validation.model_history_tracker(
    metric_names=evaluation_metric_names)

# Create an Iterative Process
iterative_process = (
      tff.learning.federated_averaging.build_federated_averaging_process(
          model_fn=lambda : model.model_fn(
              extended_vocab_size=extended_vocab_size,
              embedding_dim=EMBEDDING_DIM,
              embedding_matrix=embedding_matrix,
              rnn_units=RNN_UNITS,
              vocab_size=VOCAB_SIZE,
              sample_batch=sample_batch),
          server_optimizer_fn=federated.server_optimizer_fn,
          client_weight_fn=federated.client_weight_fn))

# Apply Transfer Learning if the Model has been Pretrained
if NUM_PRETRAINING_ROUNDS > 0:
    server_state = transfer_learning.learn_from_pretrained_model(
        iterative_process, keras_model_sp)
else:
    server_state = iterative_process.initialize()

# Train Model Across Many Randomly Sampled Clients with Federated Averaging
start_time = time.time()
for round_num in tqdm(range(0, NUM_ROUNDS)):

    # Examine validation metrics
    print('Evaluating before round #{} on {} examples.'.format(
        round_num, NUM_VALIDATION_EXAMPLES))
    validation.keras_evaluate(state=server_state,
                              val_dataset=val_data,
                              extended_vocab_size=extended_vocab_size,
                              vocab_size=VOCAB_SIZE,
                              embedding_dim=EMBEDDING_DIM,
                              embedding_matrix=embedding_matrix,
                              rnn_units=RNN_UNITS,
                              metrics_tracker=val_metrics_tracker,
                              checkpoint_dir=sav)

    # Sample train clients to create a train dataset
    print('\nSampling {} new clients.'.format(NUM_TRAIN_CLIENTS))
    train_clients = federated.get_sample_clients(
        dataset=train_data, num_clients=NUM_TRAIN_CLIENTS)
    train_datasets = [train_data.create_tf_dataset_for_client(
        client) for client in train_clients]

    # Apply federated training round
    server_state, server_metrics = iterative_process.next(
        server_state, train_datasets)

    # Add train metrics to tracker, print current value, and save
    for i, name in enumerate(train_metrics_tracker.metric_names):

        result = getattr(server_metrics, name)
        train_metrics_tracker.add_metrics_by_name(name, result)
        print('   {}: {}'.format(name, result))

        prefix = 'train_' if ('loss' in name or 'accuracy' in name) else ''
        np.save(sav + prefix + name + '.npy',
            train_metrics_tracker.get_metrics_by_name(name))

    # Write time since start of training
    with open(sav + 'train_time.txt', 'a+') as f:
        f.write('{}\n'.format(time.time() - start_time))

# Set Plot Titles Based on Training Configuration
round_config = 'Clients: {},\
    Max Elements per Client: {},\
    Max Seq Len: {},\
    Rounds: {}'.format(
        NUM_TRAIN_CLIENTS,
        MAX_ELEMENTS_PER_USER,
        MAX_SEQ_LENGTH,
        NUM_ROUNDS)

# Plot Train and Validation Loss
fig, ax = plt.subplots(figsize=(20, 15))
x_ax = range(0, NUM_ROUNDS)
ax.plot(x_ax, np.load(sav + 'train_loss.npy'), label='Train')
ax.plot(x_ax, np.load(sav + 'val_loss.npy'), label='Val')
ax.legend(loc='best', prop={'size': 15})
plt.title('Loss by Epoch - {}'.format(round_config), fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.tight_layout()
plt.savefig(sav + 'Loss by Epoch.png')

# Plot Train and Validation Accuracy
fig, ax = plt.subplots(figsize=(20, 15))
x_ax = range(0, NUM_ROUNDS)
ax.plot(x_ax, np.load(sav + 'train_accuracy_no_oov_no_eos.npy'), label='Train')
ax.plot(x_ax, np.load(sav + 'val_accuracy_no_oov_no_eos.npy'), label='Val')
ax.legend(loc='best', prop={'size': 15})
plt.title('Accuracy by Epoch - {}'.format(round_config), fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy No OOV No EOS', fontsize=18)
plt.tight_layout()
plt.savefig(sav + 'Accuracy No OOV No EOS by Epoch.png')

# Load Train Sample Stats
examples = np.load(sav + 'num_examples.npy')
tokens = np.load(sav + 'num_tokens.npy')
tokens_no_oov = np.load(sav + 'num_tokens_no_oov.npy')

# Define Function to Compute 95% Confidence Interval Errors
def mean_err(arr):
    """Compute 95% CI errors."""
    return 1.96 * np.std(arr)/np.sqrt(len(arr))

# Compute Train Sample Means and 95% Confidence Interval Errors
train_sample_stats = ['Examples', 'Tokens', 'Tokens No OOV']
means = [np.mean(examples), np.mean(tokens), np.mean(tokens_no_oov)]
errors = [mean_err(examples), mean_err(tokens), mean_err(tokens_no_oov)]

# Plot Train Sample Means
fig, ax = plt.subplots(figsize=(10, 10))
x_pos = np.arange(len(train_sample_stats))
ax.bar(x_pos, means, yerr=errors, align='center',
    alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Sample Mean with 95% Confidence Interval')
ax.set_xticks(x_pos)
ax.set_xticklabels(train_sample_stats)
ax.set_title('Train Sample Means - {}'.format(round_config))
plt.tight_layout()
plt.savefig(sav + '{} Round Train Sample Means.png'.format(NUM_ROUNDS))

# Plot Train Sample Distributions
fig, ax = plt.subplots(figsize=(10, 10))
plt.hist(examples, alpha=0.4, label='Num Examples')
plt.hist(tokens, alpha=0.4, label='Num Tokens')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Train Sample Distributions - {}'.format(round_config))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(sav + '{} Round Train Sample Distributions.png'.format(NUM_ROUNDS))
