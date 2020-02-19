"""
Prints the number of training examples sampled at each
training round with the training and validation accuracy
by pointing to the directory in which the experiment is saved.
"""

import sys
import numpy as np

if len(sys.argv) < 2:
    raise ValueError('Supply an experiment directory as an argument')

experiment_dir = sys.argv[1]

examples = np.load(experiment_dir + 'num_examples.npy')
train_accuracy = np.load(experiment_dir + 'train_accuracy.npy')
val_accuracy = np.load(experiment_dir + 'val_accuracy.npy')
n = len(examples)

for i in range(0, n):
    print('Sampled {} examples at round {}.'.format(examples[i], i))
    print('    Train accuracy = {}'.format(train_accuracy[i]))
    print('    Validation accuracy = {}'.format(val_accuracy[i])) 
