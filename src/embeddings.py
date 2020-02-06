
def load_pretrained_embeddings():
    
    embeddings_index = {}
    with open(EMBEDDING_PATH, 'r') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
            
    return embeddings_index

def create_matrix_from_pretrained_embeddings():
    
    embeddings_index = load_pretrained_embeddings()
    embedding_matrix = np.zeros((EXTENDED_VOCAB_SIZE, EMBEDDING_DIM))

    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def initialize_embedding():
    
    embedding_matrix = create_matrix_from_pretrained_embeddings()
    
    return tf.keras.initializers.Constant(embedding_matrix)
