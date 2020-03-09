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
Prints the number of training examples sampled at each federated training round
with the training and validation accuracy by pointing to the directory where
experiments are saved. Experiments from 'federated_nwp.py' will be saved with
the following directory format which should be passed as such to this script:

    '{}_{}_{}_{}_{}/'.format(
        NUM_PRETRAINING_ROUNDS,
        EMBEDDING_LAYER,
        EMBEDDING_DIM,
        RNN_UNITS,
        EMBEDDING_DIM)

Check the 'params.json' file to confirm these parameters for your run.
"""

import sys
import errno
import os.path
import numpy as np

if len(sys.argv) < 2:
    raise ValueError('Supply an experiment directory as an argument')

experiment_dir = sys.argv[1]
paths = {
    'examples': experiment_dir + 'num_examples.npy',
    'train_acc': experiment_dir + 'train_accuracy.npy',
    'train_acc_no_oov_no_eos':
        experiment_dir + 'train_accuracy_no_oov_no_eos.npy',
    'val_acc': experiment_dir + 'val_accuracy.npy',
    'val_acc_no_oov_no_eos':
        experiment_dir + 'val_accuracy_no_oov_no_eos.npy'
    }

if not np.all(np.array([os.path.exists(path) for path in paths.values()])):
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT),
            'One of: {}'.format(list(paths.values())))

examples = np.load(paths['examples'])
train_acc = np.load(paths['train_acc'])
train_acc_no_oov_no_eos = np.load(paths['train_acc_no_oov_no_eos'])
val_acc = np.load(paths['val_acc'])
val_acc_no_oov_no_eos = np.load(paths['val_acc_no_oov_no_eos'])

n = len(examples)
mx = np.max(val_acc)
mx_no = np.max(val_acc_no_oov_no_eos)
avg = np.mean(val_acc[-100:])
avg_no = np.mean(val_acc_no_oov_no_eos[-100:])

for i in range(0, n):
    print('Sampled {} examples at round {}.'.format(examples[i], i))
    print('    Train acc = {}'.format(train_acc[i]))
    print('    Train acc no oov no eos = {}'.format(train_acc_no_oov_no_eos[i]))
    print('    Val acc = {}'.format(val_acc[i]))
    print('    Val acc no oov no eos = {}'.format(val_acc_no_oov_no_eos[i]))
    print('    Max val acc = {}'.format(mx))
    print('    Max val acc no oov no eos = {}'.format(mx_no))
    print('    Average of last 100 val acc = {}'.format(avg))
    print('    Average of last 100 val acc no oov no eos = {}'.format(avg_no))
