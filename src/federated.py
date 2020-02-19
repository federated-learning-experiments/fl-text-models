import numpy as np
import tensorflow as tf

def client_weight_fn(local_outputs):

    num_tokens = tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

    return num_tokens # return 1.0 for uniform weighting

def server_optimizer_fn():

    return tf.keras.optimizers.Adam()

def get_sample_clients(dataset, num_clients):

    random_indices = np.random.choice(len(dataset.client_ids), size=num_clients, replace=False)

    return np.array(dataset.client_ids)[random_indices]
