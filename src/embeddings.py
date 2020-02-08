import io
import warnings

import numpy as np
import tensorflow as tf

def load_fasttext_embeddings(embedding_path):

    """
    Code modified from: https://fasttext.cc/docs/en/english-vectors.html.
    """

    fin = io.open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = [int(x) for x in fin.readline().split()]

    embeddings_index = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embeddings_index[tokens[0]] = np.array([float(x) for x in tokens[1:]])

    return embeddings_index

def load_glove_embeddings(embedding_path):

    """
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    """

    embeddings_index = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    return embeddings_index

def create_matrix_from_pretrained_embeddings(embedding_fn,
                                             embedding_dim,
                                             word2idx):

    """
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    Here we initialize the embedding matrix to random uniform values according
    to the random uniform initializer in tf.keras used by tf.keras.layers.Embedding.
    This way any words without embeddings are mapped according to this random
    distribution and can then be learned.
    """

    embeddings_index = embedding_fn()

    em_init_params = tf.random_uniform_initializer().get_config()
    embedding_matrix = np.random.uniform(low=em_init_params['minval'],
                                         high=em_init_params['maxval'],
                                         size=(len(word2idx), embedding_dim))

    missing_words = 0
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            missing_words += 1
        else:
            embedding_matrix[i] = embedding_vector

    warnings.warn(f'Words from dataset with no embedding: {missing_words}')

    return tf.keras.initializers.Constant(embedding_matrix)
