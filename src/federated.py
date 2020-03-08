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
This file is used to apply federated optimization and borrows from the
Tensorflow Federated Authors code located here at the time of writing:
github.com/tensorflow/federated/tree/master/tensorflow_federated/python/research
"""

import numpy as np
import tensorflow as tf

def client_weight_fn(local_outputs):
    """
    Weight loss by number of tokens from client dataset.
    Modify this function to return 1 for uniform weighting.

    Args:
      local_outputs: client dataset outputs.
    """

    num_tokens = tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

    return num_tokens

def server_optimizer_fn():
    """
    Server optimizer.
    """

    return tf.keras.optimizers.Adam()

def client_optimizer_fn():
    """
    Client optimizer.
    """

    return tf.keras.optimizers.Adam()

def get_sample_clients(dataset, num_clients):
    """
    Sample clients from a given dataset.

    Args:
      dataset: `tff.simulation.ClientData` representing Stack Overflow data.
      num_clients: int representing number of client datasets to sample.

    Return:
      `numpy.array()` of client ids.
    """


    num_samples = len(dataset.client_ids)
    random_indices = np.random.choice(
        num_samples, size=num_clients, replace=False)

    return np.array(dataset.client_ids)[random_indices]
