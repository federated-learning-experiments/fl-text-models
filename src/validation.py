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
                   rnn_units_2=None):

    keras_model = model.build_model(extended_vocab_size,
                              embedding_dim,
                              embedding_matrix,
                              rnn_units,
                              stacked_lstm=stacked_lstm,
                              rnn_units_2=rnn_units_2
                              )

    evaluation_metrics = get_metrics(vocab_size)

    model.compile_model(keras_model, evaluation_metrics)
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)

    evaluation_results = keras_model.evaluate(val_dataset)

    for i, result in enumerate(evaluation_results):
        metrics_tracker.add_metrics_by_name(metrics_tracker.metric_names[i], result)

class model_history_tracker:

    def __init__(self, metric_names=[]):

        self.metric_names = metric_names
        self.metrics_dict = {name:[] for name in metric_names}

    def get_metrics_by_name(self, metric_name):

        return self.metrics_dict[metric_name]

    def add_metrics_by_name(self, metric_name, metric_result):

        self.metrics_dict[metric_name].append(metric_result)
