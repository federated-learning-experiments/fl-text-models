import io
import warnings

import numpy as np
import tensorflow as tf

def load_fasttext_embeddings(embedding_path):

    # """
    # Loads fasttext word embeddings.
    # Code modified from: https://fasttext.cc/docs/en/english-vectors.html.
    #
    # Args:
    #   embedding_path: `str` location of embedding file.
    #
    # Returns:
    #   embedding_index: `dict` of vocab strings and embedding `numpy.array()`.
    # """

    fin = io.open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = [int(x) for x in fin.readline().split()]

    embedding_index = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embedding_index[tokens[0]] = np.array([float(x) for x in tokens[1:]])

    return embedding_index

def load_glove_embeddings(embedding_path):

    # """
    # Loads GloVe word embeddings.
    # Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    #
    # Args:
    #   embedding_path: `str` location of embedding file.
    #
    # Returns:
    #   embedding_index: `dict` of vocab strings and embedding `numpy.array()`.
    # """

    print('loading glove ems...')
    embedding_index = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_index[word] = coefs

    print('end.')
    return embedding_index

def create_matrix_from_pretrained_embeddings(embedding_fn,
                                             embedding_path,
                                             embedding_dim,
                                             word2idx):

    # """
    # Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    # Here we initialize the embedding matrix to random uniform values according
    # to the random uniform initializer in tf.keras used by tf.keras.layers.Embedding.
    # This way any words without embeddings are mapped according to this random
    # distribution and can then be learned.
    #
    # Args:
    #   embedding_fn: either `load_glove_embeddings`
    #     or `load_fasttext_embeddings`.
    #   embedding_path: `str` location of embedding file.
    #   embedding_dim: size of each word embedding as `int`.
    #   word2idx: `dict` mapping word strings to integer indices.
    #
    # Returns:
    #   embedding_matrix: matrix of type `tf.keras.initializers.Constant`
    #     to be passed to `tf.keras.layers.Embedding`.
    # """

    embedding_index = embedding_fn(embedding_path)
    print('here')

    em_init_params = tf.random_uniform_initializer().get_config()
    embedding_matrix = np.random.uniform(low=em_init_params['minval'],
                                         high=em_init_params['maxval'],
                                         size=(len(word2idx), embedding_dim))

    missing_words = 0
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is None:
            missing_words += 1
        else:
            embedding_matrix[i] = embedding_vector

    warnings.warn(f'Words from dataset with no embedding: {missing_words}')
    embedding_matrix = tf.keras.initializers.Constant(embedding_matrix)

    return embedding_matrix
