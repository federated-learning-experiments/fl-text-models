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
This file is used to apply transfer learning from a pretrained keras
model to a federated keras model.
"""

import requests
import zipfile
import io

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from . import dataset

def learn_from_pretrained_model(iterative_process, pretrained_model):
    """
    All of the values of the pre-trained model's trainable weights
    are transferred to the TFF model to be fine-tuned.
    """
    model_to_finetune = iterative_process.initialize()
    for l1 in range(len(model_to_finetune.model[0])):
        for l2 in range(len(model_to_finetune.model[0][l1])):
            model_to_finetune.model[0][l1][l2] = pretrained_model.\
                trainable_weights[l1].numpy()[l2]

    return model_to_finetune

def load_and_preprocess_shakespeare(vocab_size):
    """
    Loads and preprocesses the raw shakespeare dataset from project Gutenburg.

    Args:
      vocab_size: Integer representing size of the vocab to use. Voca will
        then be the `vocab_size` most frequent words in the Shakespeare dataset.
    Returns:
      (X, Y, vocab_sp): Where X and Y are sequences of
        type `np.array` where X is each vocab word and Y is
        offset to be the next word in the sequence.  The last item in the
        tuple is the list of most frequent words in the Shakespeare dataset,
        not including special tokens.
    """

    r = requests.get('http://www.gutenberg.org/files/100/old/1994-01-100.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('../')

    with open('../100.txt') as f:
        data = f.read()

    word_counts_sp = {}
    for word in data.split():
        word_counts_sp[word] = word_counts_sp.get(word, 0) + 1

    sorted_word_counts_sp = {k: v for k, v in sorted(word_counts_sp.items(),
                            key=lambda item: item[1], reverse=True)}

    vocab_sp = list(sorted_word_counts_sp.keys())[:vocab_size]

    pad, oov, bos, eos = dataset.get_special_token_words()
    extended_vocab = [pad] + vocab_sp + [oov, bos, eos]
    word2idx = {word: i for i, word in enumerate(extended_vocab)}

    encoded = []
    for word in data.split():
        if not word2idx.get(word, None): # return None if not found
            encoded.append(0) # 0 is the pad token
        else:
            encoded.append(word2idx[word])

    X, Y = [], []
    for i in range(len(encoded)-1):
        X.append(encoded[i])
        Y.append(encoded[i + 1])

    X = np.expand_dims(X, 1)
    Y = np.expand_dims(Y, 1)

    return X, Y, vocab_sp
