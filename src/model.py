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
This file is used to build federated keras models.
"""

import tensorflow as tf
import tensorflow_federated as tff

import os, sys
sys.path.append(os.getcwd())

from . import validation

def build_model(extended_vocab_size,
                embedding_dim,
                embedding_matrix,
                rnn_units,
                stacked_lstm=False,
                rnn_units_2=None):
    """
    Build model with architecture
    from: https://www.tensorflow.org/tutorials/text/text_generation.
    """
    model1_input = tf.keras.Input(shape=(None, ),
                                  name='model1_input')

    model1_embedding = tf.keras.layers.Embedding(
        input_dim=extended_vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=embedding_matrix,
        mask_zero=True,
        trainable=True,
        name='model1_embedding')(model1_input)

    model1_lstm = tf.keras.layers.LSTM(
        units=rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        name='model1_lstm')(model1_embedding)

    if stacked_lstm:
        model2_lstm = tf.keras.layers.LSTM(
            units=rnn_units_2,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            name='model2_lstm')(model1_lstm)

        model_lstm = model2_lstm
    else:
        model_lstm = model1_lstm

    dense1 = tf.keras.layers.Dense(units=embedding_dim)(model_lstm)

    dense2 = tf.keras.layers.Dense(units=extended_vocab_size)(dense1)

    final_model = tf.keras.Model(inputs=model1_input, outputs=dense2)

    return final_model

def compile_model(keras_model, evaluation_metrics):
    """
    Compile a given keras model using SparseCategoricalCrossentropy
    loss and the Adam optimizer with set evaluation metrics.
    """

    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=evaluation_metrics)

    return keras_model

def model_fn(extended_vocab_size,
             embedding_dim,
             embedding_matrix,
             rnn_units,
             vocab_size,
             sample_batch,
             stacked_lstm=False,
             rnn_units_2=None):
    """
    Create TFF model from compiled Keras model and a sample batch.
    """

    keras_model = build_model(extended_vocab_size,
                              embedding_dim,
                              embedding_matrix,
                              rnn_units,
                              stacked_lstm=stacked_lstm,
                              rnn_units_2=rnn_units_2)

    evaluation_metrics = validation.get_metrics(vocab_size)

    compile_model(keras_model, evaluation_metrics)

    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)
