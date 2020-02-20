import os, sys
sys.path.append(os.getcwd())

from . import dataset

import warnings
import numpy as np
import tensorflow as tf

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.decomposition import PCA

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
      word2embedding: `dict` of vocab strings and embeddings `numpy.array()`.
    """

    word2embedding = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            word2embedding[word] = coefs

    return word2embedding

def create_matrix_from_pretrained_embeddings(word2embedding,
                                             embedding_dim,
                                             word2idx):

    """
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    Here we initialize the embedding matrix to random uniform values according
    to the random uniform initializer in tf.keras used by tf.keras.layers.Embedding.
    This way any words without embeddings are mapped according to this random
    distribution and can be learned.  Similarly, special token words from
    `dataset.py` are mapped to random values to be learned and not mapped to
    word embeddings in case embeddings exist for words like 'pad'.

    Args:
      word2embedding: `dict` of vocab strings and embeddings `numpy.array()`.
      embedding_dim: size of each word embedding as `int`.
      word2idx: `dict` mapping word strings to integer indices.

    Returns:
      embedding_matrix: matrix of type `tf.keras.initializers.Constant`
        to be passed to `tf.keras.layers.Embedding`.
    """

    special_token_words = list(dataset.get_special_token_words())

    em_init_params = tf.random_uniform_initializer().get_config()
    embedding_matrix = np.random.uniform(low=em_init_params['minval'],
                                         high=em_init_params['maxval'],
                                         size=(len(word2idx), embedding_dim))

    missing = 0
    for word, i in word2idx.items():
        vector = word2embedding.get(word)
        if (vector is None) or (word in special_token_words):
            missing += 1
        else:
            embedding_matrix[i] = vector

    warnings.warn('{} words set to default random initialization'\
            .format(missing), missing_words_warning)

    return tf.keras.initializers.Constant(embedding_matrix)

def create_gpt_embeddings(word2idx):
    """
    Code modified from: https://github.com/huggingface/transformers/issues/1458.
    Here we extract the word embeddings from the gpt model in the huggingface
    transformers library and return the embedding index dict.

    Args:
      word2idx: `dict` mapping word strings to integer indices.

    Returns:
      word2embedding: `dict` of vocab strings and embeddings `numpy.array()`.
    """

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    word2embedding = {}

    for word, _ in word2idx.items():
        text_index = tokenizer.encode(word)
        vector = model.transformer.wte.weight[text_index,:]
        word2embedding[word] = vector[0].detach().numpy()

    return word2embedding

def to_pca_projections(word2embedding, n):
    """
    Takes an embedding index dictionary with numpy arrays as keys
    and applys PCA to the key vectors returning a new embedding
    index dictionary based on the top n principle components.

    Args:
      word2embedding: `dict` of vocab strings and embeddings `numpy.array()`.
      n: int representing number of principle components to use for projection

    Returns:
      word2embedding: `dict` of vocab strings and embeddings `numpy.array()`.
    """

    words, vectors = [], []
    for word, vector in word2embedding.items():
        words.append(word)
        vectors.append(vector)

    X = np.array(vectors)
    X_scaled = X - X.mean(axis=0)

    pca = PCA(n_components=n)
    X_projected = pca.fit_transform(X_scaled)

    word2embedding = {}
    for i, vector in enumerate(X_projected):
        word2embedding[words[i]] = vector

    return word2embedding
