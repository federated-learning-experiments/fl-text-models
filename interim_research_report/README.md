# Interim Research Report
- [Joel Stremmel\*](https://github.com/jstremme)
- [Arjun Singh\*](https://github.com/sinarj)

### Motivation
While machine learning on large datasets is the dominant paradigm in the field, there are a number of drawbacks to centrally aggregating data, namely privacy.  Federated Learning aims to address this and has shown promise for text completion tasks on mobile devices. The [Tensorflow Federated API](https://github.com/tensorflow/federated) provides methods to train Federated models and conduct Federated Learning experiments on data grouped by clients but never aggregated.  Through our research partnership with Google, we aim to build on the existing body of Federated Learning experiments with a particular focus on enhancing text models for Natural Language Understanding tasks, such as Next Word Prediction.

### Problem Statement
Federated Learning aims to train machine learning models in a distributed fashion without centralizing data but instead updating and passing model parameters from a central server to distributed entities and back to perform stochastic gradient descent.  McMahan et al. propose the Federated Averaging algorithm in Communication-efficient learning of deep networks from decentralized data.  Our goal is to replicate the existing network architectures for Federated Averaging, stress testing their limits within our simulated environment in terms of compute, memory and power resources. We then aim to apply pretraining and pretrained model layers to measure the impact of starting with learned models weights compared to random initialization for the tast of Next Word Prediction on Stack Overflow.  Specifically, in this interim delivery, we measure the effect of using pretrained embeddings on the number of training rounds required to achieve a fixed level of accuracy.

### Data
The main dataset used for these experiments is hosted by Kaggle and made available through the [tff.simulation.datasets module in the Tensorflow Federated API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data).  Stack Overflow owns the data and has released the data under the [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).  The Stack Overflow data contains the full body text of all Stack Overflow questions and answers along with metadata with the API pointer updated quarterly.  The data is split into the following sets:

- Train: 342,477 distinct users and 135,818,730 examples.
- Validation: 38,758 distinct users and 16,491,230 examples.
- Test: 204,088 distinct users and 16,586,035 examples.

The following notebook contains an (exploratory analysis)[https://github.com/federated-learning-experiments/fl-text-models/blob/master/local_gpu_training/eda/stack_overflow_eda.ipynb] of the data with example records and visualizations.  From this notebook we deduce that challenges with the data include:
- The size of the data, as it would be nearly impossible to inspect all samples.
- The distribution of words.  As is common with text data, the most common words occur with frequency far greater than the least common words (see [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law)).  Therefore, in our experiments, we limit the vocab size to exclude very rare words, accepting that even state of the art language models fail at next word prediction when the next word is rare.

### Model Design
For this interim delivery, we train two neural networks with four layers each and compare train and validation accuracy at each training round for 500 training rounds by sampling 10 training client datasets per round each with 5000 non-IID text samples from Stack Overflow at maximum and a total of 10000 validation text samples.  Each of the two models are trained with the Federated Averaging Algorithm as in [McMahan et. al.](https://arxiv.org/pdf/1602.05629.pdf).  The model architecure is as follows:

![plot1](network)

Note in the above that one network starts with a randomly initialized embedding layer while the other starts with [pretrained GloVe embeddings](https://nlp.stanford.edu/projects/glove/) trained on the Wikipedia2015 + Gigaword5 text corpus.  A majority but not all of the words in the Stack Overflow vocabulary have corresponding GloVe embeddings.  For this reason, we set this layer to trainable to learn embeddings for words without GloVe representations and fine tune embeddings with existing representations.

### Results
With the model design for the two networks fixed with the exception of the starting embedding layers, we train the model using the Adam optimizer and Sparse Categorical Crossentropy loss, measuring accuracy including out of vocab and end of sentence tokens after each training round on both the sampled training client datasets and the fixed validation set.

**Model Objective Function at Each Training Round**
Note that epochs and training rounds are equivalent as we apply federated averaging after each round as opposed to applying optimization steps on each client dataset for multiple epochs in between training rounds.
![plot2](loss)

**Accuracy at Each Training Round**
Note that epochs and training rounds are equivalent as we apply federated averaging after each round as opposed to applying optimization steps on each client dataset for multiple epochs in between training rounds.
![plot3](accuracy)

### Discussion and Next Steps








