import tensorflow as tf
import tensorflow_federated as tff

from validation import get_metrics

def build_model(vocab_size,
                embedding_dim,
                embedding_matrix,
                rnn_units):
    """
    Build model with architecture from: https://www.tensorflow.org/tutorials/text/text_generation.
    """

    model1_input = tf.keras.Input(shape=(None, ),
                                  name='model1_input')

    model1_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                 output_dim=embedding_dim,
                                                 embeddings_initializer=embedding_matrix,
                                                 mask_zero=True,
                                                 trainable=True,
                                                 name='model1_embedding')(model1_input)

    model1_lstm = tf.keras.layers.LSTM(units=rnn_units,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform',
                                       name='model1_lstm')(model1_embedding)

    model1_dense1 = tf.keras.layers.Dense(units=embedding_dim)(model1_lstm)

    model1_dense2 = tf.keras.layers.Dense(units=vocab_size)(model1_dense1)

    final_model = tf.keras.Model(inputs=model1_input, outputs=model1_dense2)

    return final_model

def compile_model(keras_model, evaluation_metrics):

    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=evaluation_metrics)

    return keras_model

def model_fn(vocab_size,
             embedding_dim,
             embedding_matrix,
             rnn_units,
             sample_batch):
    """
    Create TFF model from compiled Keras model and a sample batch.
    """

    keras_model = build_model(vocab_size,
                              embedding_dim,
                              embedding_matrix,
                              rnn_units)

    evaluation_metrics = get_metrics()

    compile_model(keras_model, evaluation_metrics)

    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)
