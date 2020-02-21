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
    random_indices = np.random.choice(num_samples, size=num_clients, replace=False)

    return np.array(dataset.client_ids)[random_indices]
