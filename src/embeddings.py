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
This file is used to construct word embeddings for federated models.
"""

import os, sys
sys.path.append(os.getcwd())

from . import dataset

import warnings
import errno

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

class missing_words_warning(UserWarning):
    """
    Warning regarding words missing from word to embedding mapping.
    """
    pass

def load_embeddings(embedding_path):
    """
    Loads word embeddings.
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.

    Args:
      embedding_path: `str` location of embedding file.

    Returns:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
    """

    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), embedding_path)

    word2embedding = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            if coefs.shape[0] > 1:
                word2embedding[word] = coefs

    return word2embedding

def create_matrix_from_pretrained_embeddings(word2embedding,
                                             embedding_dim,
                                             vocab):
    """
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    Here we initialize the embedding matrix to random uniform values according
    to the random uniform initializer in tf.keras used by
    `tf.keras.layers.Embedding`. This way any words without embeddings are
    mapped according to this random distribution and can be learned.

    Args:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
      embedding_dim: size of each word embedding as `int`.
      vocab: `list` of words in vocab.

    Returns:
      embedding_matrix: matrix of type `tf.keras.initializers.Constant`
        to be passed to `tf.keras.layers.Embedding`.
    """

    # Extended vocab includes tokens in the order of preprocessing in dataset.py
    pad, oov, bos, eos = dataset.get_special_token_words()
    extended_vocab = [pad] + vocab + [oov, bos, eos]

    init_params = tf.random_uniform_initializer().get_config()
    embedding_matrix = np.random.uniform(low=init_params['minval'],
                                         high=init_params['maxval'],
                                         size=(len(extended_vocab),
                                               embedding_dim))

    missing = 0
    for i, word in enumerate(extended_vocab):
        vector = word2embedding.get(word)
        if (word not in [pad, oov, bos, eos]) and (vector is not None):
            embedding_matrix[i] = vector
        else:
            missing += 1

    warnings.warn('{} words set to default random initialization'\
            .format(missing), missing_words_warning)

    return tf.keras.initializers.Constant(embedding_matrix)

def create_gpt_embeddings(vocab):
    """
    Code modified from: https://github.com/huggingface/transformers/issues/1458.
    Here we extract the word embeddings from the gpt model in the huggingface
    transformers library and return the embedding index dict.

    Args:
      vocab: `list` of words in vocab.

    Returns:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
    """

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    word2embedding = {}

    for word in vocab:
        text_index = tokenizer.encode(word)
        if text_index:
            vector = model.transformer.wte.weight[text_index,:]
            word2embedding[word] = vector[0].detach().numpy()

    return word2embedding

def to_vocab_words_and_vectors(word2embedding, vocab):
    """
    Takes a dictionary of words and vectors along with a vocabulary
    list and returns the words and vectors in the vocabulary
    as a (words, vectors) tuple.

    Args:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
      vocab: `list` of words in vocab.

    Returns:
      words, vectors: tuple with elements of type `list`.
    """

    w2e_in_vocab = {w:v for w, v in word2embedding.items() if w in vocab}

    words, vectors = [], []
    for word, vector in w2e_in_vocab.items():
        words.append(word)
        vectors.append(vector)

    return words, vectors

def to_pca_projections(word2embedding, vocab, n):
    """
    Takes a dictionary of words and vectors along with a vocabulary
    list and applys PCA to the in vocab word vectors returning a new embedding
    word embedding dictionary based on the top n principle components.

    Args:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
      vocab: `list` of words in vocab.
      n: `int` representing number of principle components to use for projection.

    Returns:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
    """

    words, vectors = to_vocab_words_and_vectors(word2embedding, vocab)
    X = np.array(vectors)
    X_train = X - np.mean(X, axis=0)

    pca = PCA(n_components=n)
    X_projected = pca.fit_transform(X_train)

    word2embedding = {}
    for i, vector in enumerate(X_projected):
        word2embedding[words[i]] = vector

    return word2embedding

def plot_pca_variance_explained(word2embedding, vocab):
    """
    Takes a dictionary of words and vectors along with a vocabulary
    list and applys PCA to the in vocab word vectors for all possible
    numbers of components and plots the variance explained by each.

    Args:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
      vocab: `list` of words in vocab.
    """

    _, vectors = to_vocab_words_and_vectors(word2embedding, vocab)
    X = np.array(vectors)
    X_train = X - np.mean(X, axis=0)

    pca_full = PCA(n_components=min(X_train.shape[0], X_train.shape[1]))
    pca_full.fit(X_train)

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Variance Explained by Principal Components')
    plt.show()

def to_pp_pca_pp_projections(word2embedding, vocab, n, d=7):
    """
    Takes a dictionary of words and vectors along with a vocabulary
    list and applys the pre and post processing PCA algorithm from
    Vikas Raunak's 2017 NIPS paper: https://arxiv.org/pdf/1708.03629.pdf
    for a desired embedding dimension n and the top d dimensions parameter
    from the paper. Implementation based on code from
    https://github.com/vyraun/Half-Size.

    Args:
      word2embedding: `dict` of words and embeddings `numpy.array()`.
      vocab: `list` of words in vocab.
      n: `int` representing num of principle components to use for projection.
      d: `int` top d dimensions parameter from the paper which used d=7.

    Returns:
      final_pca_embeddings: `dict` of words and embeddings `numpy.array()`.
    """

    words, vectors = to_vocab_words_and_vectors(word2embedding, vocab)
    X = np.array(vectors)
    X_train = X - np.mean(X, axis=0)

    # PCA to get Top Components
    pca =  PCA(n_components=X_train.shape[1])
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    # Removing Projections on Top Components
    z = []
    for i, x in enumerate(X_train):
    	for u in U1[0:d]:
            x = x - np.dot(u.transpose(), x) * u
    	z.append(x)

    # PCA Dim Reduction
    z = np.array(z)
    X_train = z - np.mean(z, axis=0)

    pca =  PCA(n_components=n)
    X_new_final = pca.fit_transform(X_train)

    # PCA to do Post-Processing Again
    X_new = X_new_final - np.mean(X_new_final, axis=0)

    pca =  PCA(n_components=n)
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    X_new_final = X_new_final - np.mean(X_new_final, axis=0)

    final_pca_embeddings = {}
    for i, word in enumerate(words):
        final_pca_embeddings[word] = X_new_final[i]
        for u in Ufit[0:d]:
            final_pca_embeddings[word] = final_pca_embeddings[word] - \
                np.dot(u.transpose(), final_pca_embeddings[word]) * u

    return final_pca_embeddings

def build_embedding_layer(embedding_type, embedding_dim, vocab,
        base_path='word_embeddings/'):
    """
    If the 'embedding_type' option is set to 'random',
    the embedding matrix in the embedding layer is
    initialized according to the random uniform distribution
    used by the `tf.keras` embedding layer by passing the
    'uniform' string as an argument to the 'embedding_initializer'
    in the `model.build_model` function.

    If one of the allowed word embedding options is set,
    an index called 'word2embedding' is created from pretrained
    embeddings either loaded from the 'word_embeddings' directory
    which must be created within the 'fl-text-models' directory
    or created from a pretrained model in the case of GPT2 embeddings.
    The 'word2embedding' dictionary is used to create a
    `tf.keras.initializers.Constant` embedding matrix to be
    passed to the `model.build_model` function.

    If the the 'embedding_type' option is not set to one
    of the allowed options or the GloVe or FastText option
    is set when the 'word_embeddings'' directory does not exist,
    this function will raise an error.

    Args:
      embedding_type: `str` equal to one of:
                   ['random',
                    'glove',
                    'fasttext',
                    'pca_fasttext',
                    'pp_pca_pp_fasttext',
                    'gpt2',
                    'pca_gpt2',
                    'pp_pca_pp_gpt2'].
      embedding_dim: size of each word embedding as `int`.
      vocab: `list` of words in vocab.
      base_path: `str` path to the `word_embedding` dir.

    Returns:
      embedding_matrix: matrix of type `tf.keras.initializers.Constant`
        to be passed to `tf.keras.layers.Embedding`.
    """

    if embedding_type == 'random':
        pass

    elif embedding_type == 'glove':
        embedding_path = base_path + 'glove/glove.6B.{}d.txt'.format(
            embedding_dim)
        word2embedding = load_embeddings(embedding_path)

    elif embedding_type == 'fasttext':
        embedding_path = base_path + 'fasttext/wiki-news-300d-1M.vec'
        word2embedding = load_embeddings(embedding_path)

    elif embedding_type == 'pca_fasttext':
        embedding_path = base_path + 'fasttext/wiki-news-300d-1M.vec'
        word2embedding = load_embeddings(embedding_path)
        word2embedding = to_pca_projections(
            word2embedding, vocab, embedding_dim)

    elif embedding_type == 'pp_pca_pp_fasttext':
        embedding_path = base_path + 'fasttext/wiki-news-300d-1M.vec'
        word2embedding = load_embeddings(embedding_path)
        word2embedding = to_pp_pca_pp_projections(
            word2embedding, vocab, embedding_dim)

    elif embedding_type == 'gpt2':
        word2embedding = create_gpt_embeddings(vocab)
        word2embedding = to_pca_projections(
            word2embedding, vocab, embedding_dim)

    elif embedding_type == 'pca_gpt2':
        word2embedding = create_gpt_embeddings(vocab)
        word2embedding = to_pca_projections(
            word2embedding, vocab, embedding_dim)

    elif embedding_type == 'pp_pca_pp_gpt2':
        word2embedding = create_gpt_embeddings(vocab)
        word2embedding = to_pp_pca_pp_projections(
            word2embedding, vocab, embedding_dim)

    else:
        layer_opts = ['random',
                      'glove',
                      'fasttext',
                      'pca_fasttext',
                      'pp_pca_pp_fasttext',
                      'gpt2',
                      'pca_gpt2',
                      'pp_pca_pp_gpt2']

        raise ValueError("Embedding layer type must be in {}.".format(
            layer_opts))

    if embedding_type == 'random':
        embedding_matrix = 'uniform'
    else:
        embedding_matrix = create_matrix_from_pretrained_embeddings(
            word2embedding=word2embedding,
            embedding_dim=embedding_dim,
            vocab=vocab)

    return embedding_matrix
