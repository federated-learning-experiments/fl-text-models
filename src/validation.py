import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

import os, sys
sys.path.append(os.getcwd())

from . import dataset, metrics, model

def get_metrics(vocab_size):

    pad, oov, _, eos = dataset.get_special_tokens(vocab_size)

    evaluation_metrics = [
        metrics.NumTokensCounter(name='num_tokens', masked_tokens=[pad]),
        metrics.NumTokensCounter(name='num_tokens_no_oov', masked_tokens=[pad, oov]),
        metrics.NumBatchesCounter(name='num_batches'),
        metrics.NumExamplesCounter(name='num_examples'),
        metrics.MaskedCategoricalAccuracy(name='accuracy', masked_tokens=[pad]),
        metrics.MaskedCategoricalAccuracy(name='accuracy_no_oov', masked_tokens=[pad, oov]),
        metrics.MaskedCategoricalAccuracy(name='accuracy_no_oov_no_eos', masked_tokens=[pad, oov, eos])
    ]

    return evaluation_metrics

def keras_evaluate(state,
                   val_dataset,
                   extended_vocab_size,
                   vocab_size,
                   embedding_dim,
                   embedding_matrix,
                   rnn_units,
                   metrics_tracker,
                   stacked_lstm=False,
                   rnn_units_2=None,
                   checkpoint_dir=None):

    # initalized weights will be replaced with weights from tff model state
    keras_model = model.build_model(extended_vocab_size,
                                    embedding_dim,
                                    embedding_matrix,
                                    rnn_units,
                                    stacked_lstm=stacked_lstm,
                                    rnn_units_2=rnn_units_2)

    evaluation_metrics = get_metrics(vocab_size)

    model.compile_model(keras_model, evaluation_metrics)
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)

    evaluation_results = keras_model.evaluate(val_dataset)

    for i, result in enumerate(evaluation_results):
        metrics_tracker.add_metrics_by_name(metrics_tracker.metric_names[i], result)

    if checkpoint_dir:
        metric_values = metrics_tracker.get_metrics_by_name(metrics_tracker.champion_metric_name)
        current_iter = len(metric_values) - 1

        if current_iter == metrics_tracker.champion_metric_iter:
            print('\nSaving model weights at iteration: {}'.format(current_iter))
            keras_model.save_weights(filepath=checkpoint_dir + 'weights.h5', overwrite=True)

def load_and_test_model_from_checkpoint(checkpoint,
                                        test_dataset,
                                        extended_vocab_size,
                                        vocab_size,
                                        embedding_dim,
                                        rnn_units,
                                        metrics_tracker,
                                        stacked_lstm=False,
                                        rnn_units_2=None):

    # initalized weights will be replaced with weights from checkpoint
    keras_model = model.build_model(extended_vocab_size=extended_vocab_size,
                                    embedding_dim=embedding_dim,
                                    embedding_matrix='uniform',
                                    rnn_units=rnn_units,
                                    stacked_lstm=stacked_lstm,
                                    rnn_units_2=rnn_units_2)

    keras_model.load_weights(checkpoint)
    evaluation_metrics = get_metrics(vocab_size)

    model.compile_model(keras_model, evaluation_metrics)
    sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(test_dataset)))
    tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    evaluation_results = keras_model.evaluate(test_dataset)

    metrics, results = [], []
    with open(checkpoint + '_test_metrics.txt', 'w') as f:
        for i, result in enumerate(evaluation_results):
            f.write('{}, {}\n'.format(metrics_tracker.metric_names[i], result))
            print(metrics_tracker.metric_names[i], result)
            metrics.append(metrics_tracker.metric_names[i])
            results.append(result)

    return metrics, results

class model_history_tracker:

    def __init__(self,
                 metric_names=[],
                 champion_metric_name='accuracy',
                 champion_metric_iter=0,
                 champion_metric_value=0):

        self.metric_names = metric_names
        self.champion_metric_name = champion_metric_name
        self.champion_metric_iter = champion_metric_iter
        self.champion_metric_value = champion_metric_value
        self.metrics_dict = {name:[] for name in metric_names}

    def get_metrics_by_name(self, metric_name):

        return self.metrics_dict[metric_name]

    def add_metrics_by_name(self, metric_name, metric_result):

        # add the new metrics
        self.metrics_dict[metric_name].append(metric_result)

        # update the champion metric
        if metric_name == self.champion_metric_name:
            metric_values = np.array(self.metrics_dict[metric_name])
            self.champion_metric_iter = np.argmax(metric_values)
            self.champion_metric_value = metric_values[self.champion_metric_iter]
