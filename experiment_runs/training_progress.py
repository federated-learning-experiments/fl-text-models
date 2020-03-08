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
paths = {'examples': experiment_dir + 'num_examples.npy',
         'train_acc': experiment_dir + 'train_accuracy.npy',
         'val_acc': experiment_dir + 'val_accuracy.npy'}

if not np.all(np.array([os.path.exists(path) for path in paths.values()])):
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT),
            'One of: {}'.format(list(paths.values())))

examples = np.load(paths['examples'])
train_accuracy = np.load(paths['train_acc'])
val_accuracy = np.load(paths['val_acc'])
n = len(examples)

for i in range(0, n):
    print('Sampled {} examples at round {}.'.format(examples[i], i))
    print('    Train accuracy = {}'.format(train_accuracy[i]))
    print('    Validation accuracy = {}'.format(val_accuracy[i]))
    print('    Max validation accuracy = {}'.format(np.max(val_accuracy)))
