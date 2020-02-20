import warnings

import numpy as np
import tensorflow as tf

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_embeddings(embedding_path):

    """
    Loads word embeddings.
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.

    Args:
      embedding_path: `str` location of embedding file.

    Returns:
      embedding_index: `dict` of vocab strings and embedding `numpy.array()`.
    """

    embedding_index = {}
    with open(embedding_path, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_index[word] = coefs

    return embedding_index

def create_matrix_from_pretrained_embeddings(embedding_path,
                                             embedding_dim,
                                             word2idx):

    """
    Code modified from: https://keras.io/examples/pretrained_word_embeddings/.
    Here we initialize the embedding matrix to random uniform values according
    to the random uniform initializer in tf.keras used by tf.keras.layers.Embedding.
    This way any words without embeddings are mapped according to this random
    distribution and can then be learned.

    Args:
      embedding_path: `str` location of embedding file.
      embedding_dim: size of each word embedding as `int`.
      word2idx: `dict` mapping word strings to integer indices.

    Returns:
      embedding_matrix: matrix of type `tf.keras.initializers.Constant`
        to be passed to `tf.keras.layers.Embedding`.
    """

    embedding_index = load_embeddings(embedding_path)

    em_init_params = tf.random_uniform_initializer().get_config()
    embedding_matrix = np.random.uniform(low=em_init_params['minval'],
                                         high=em_init_params['maxval'],
                                         size=(len(word2idx)+1, embedding_dim))

    missing_words = 0
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is None:
            missing_words += 1
        else:
            embedding_matrix[i] = embedding_vector

    warnings.warn('Words from dataset with no embedding: {}'.format(missing_words))
    embedding_matrix = tf.keras.initializers.Constant(embedding_matrix)

    return embedding_matrix

def create_gpt_embeddings(word2idx):

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    embedding_index = {}

    for word, _ in word2idx.items():
        text_index = tokenizer.encode(word)
        print(type_text_index)
        vector = model.transformer.wte.weight[text_index,:]
        print(type(vector))
        embedding_index[word] = vector[0].detach().numpy()

    return embedding_index