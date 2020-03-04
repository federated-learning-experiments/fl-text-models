# Final Research Report
##### DATA 591 at University of Washington in Collaboration with Google

- 03-11-2020
- [Arjun Singh\*](https://github.com/sinarj), [Joel Stremmel\*](https://github.com/jstremme)
- Special thanks to Keith Rush and Peter Kairouz from Google for their guidance throughout the course of this project.

### Introduction
While training on centralized data is the dominant paradigm in machine learning, there are a variety of limitations to centralized model training, such as compromised user privacy and maintenance of expensive compute resources.  Federated learning aims to address these challenges and has exhibited promising results for [text completion tasks on mobile devices](https://arxiv.org/pdf/1811.03604.pdf). The [Tensorflow Federated API](https://github.com/tensorflow/federated) is a programming interface for training federated models and conducting federated learning experiments on data grouped by individual clients but never centrally aggregated.  Through our research partnership with the Tensorflow Federated team at Google, we build on the existing body of federated learning experiments, focussing on enhancing the accuracy and reducing the size and training time constraints of federated text models for next word prediction.

### Enhancing Federated Text Models with Pretraining Methods
Federated learning trains machine learning models in a distributed fashion without centralizing data but instead updating and passing model parameters from a central server to distributed entities and back to perform stochastic gradient descent.  McMahan et al. propose the Federated Averaging algorithm in ["Communication-Efficient Learning of Deep Networks from Decentralized Data."](https://arxiv.org/pdf/1602.05629.pdf)  In this research we replicate a baseline network architecture for next word prediction using the Federated Averaging algorithm to train an LSTM on the Stack Overflow dataset, then apply three enhancements to federated training with this architecture, demonstrating increased accuracy with fewer required training rounds.  Our enhancements include 1) centrally pretraining deep neural network models then fine tuning them in the federated setting, 2) incorporating pretrained word embeddings instead of randomly initialized embeddings and fine tuning these embeddings while training the full network in the federated setting, and 3) combining centralized pretraining and pretrained word embeddings with federated fine tuning.  The following sections detail the methods we apply to demonstrate these enhancements as well as our experimental results.  All code for this research is freely available under the MIT License [at this address](https://github.com/federated-learning-experiments/fl-text-models).

### Data
The main dataset used for these experiments is hosted by Kaggle and made available through the [tff.simulation.datasets module in the Tensorflow Federated API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data).  Stack Overflow owns the data and has released the data under the [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).  The Stack Overflow data contains the full body text of all Stack Overflow questions and answers along with metadata, and the API pointer is updated quarterly.  The data is split into the following sets at the time of writing:

- Train: 342,477 distinct users and 135,818,730 examples.
- Validation: 38,758 distinct users and 16,491,230 examples.
- Test: 204,088 distinct users and 16,586,035 examples.

The [EDA notebook linked here](https://github.com/federated-learning-experiments/fl-text-models/blob/master/eda/stack_overflow_eda.ipynb) contains an exploratory analysis of the data with example records and visualizations.  From this notebook we deduce that challenges with the data include:

- The size of the data, as it would be nearly impossible to inspect all samples.
- The distribution of words.  As is common with text data, the most common words occur with frequency far greater than the least common words (see [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law)).  

Therefore, in our experiments, we limit the vocab size to exclude very rare words, accepting that even state of the art language models fail at next word prediction when the next word is rare.

### Model Design
In this study, we train a variety of small and large neural networks with four layers each and compare train and validation accuracy at each training round for 800 training rounds by sampling 10 training client datasets per round, each with 5,000 non-IID text samples from Stack Overflow at maximum, and a total of 20,000 validation text samples.  While training each network, we save the model weights from the training round that produced the highest validation set accuracy and conduct a final performance evaluation on 1,000,000 test samples. All models are trained with the Federated Averaging algorithm as in [McMahan et. al.](https://arxiv.org/pdf/1602.05629.pdf) using the Tensorflow Federated simulated training environment.  The large network outperforms the smaller network though has about three times the number of trainable parameters (7,831,328, 2,402,072) and is about three times as big (31.3MB vs 9.6MB).  The large network is depicted here:

![](images/model_architecture.tmp)

### Central Pretraining with Federated Fine Tuning

### Pretrained Word Embeddings for Federated Training
We hypothesize that having a common, starting representation for words across federated (non-IID) datasets yields improved model performance with fewer training rounds compared to federated training with randomly initialized word embeddings.

Results:

1. We observe an increase of over a half percent accuracy with pretrained compared to random embeddings for the larger networks when validating on 1,000,000 test samples with little to no improvement from pretrained embeddings for the smaller networks.

2. For both small and large networks we observe from the model learning curves that pretrained word embeddings achieve the same level of accuracy sooner, that is, with fewer training rounds compared to random embeddings.

### Federated Fine Tuning Using a Pretrained Model with Pretrained Word Embeddings

### Conclusions and Future Work

### References
This project draws mainly from the following research, but other sources are referenced throughout this repository, particularly code snippets.

- Tensorflow Federated [API](https://github.com/tensorflow/federated).
-	H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. [“Communication-Efﬁcient Learning of Deep Networks."](https://arxiv.org/pdf/1602.05629.pdf) Accessed December 6, 2019.
- Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander. [“Towards Federated Learning at Scale: System Design.”](https://arxiv.org/pdf/1902.01046.pdf) Accessed December 6, 2019.
- Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy, Francoise Beaufays Sean Augenstein, Hubert Eichner, Chloe Kiddon, Daniel Ramage. [“Federated Learning for Mobile Keyboard Prediction.”](https://arxiv.org/pdf/1811.03604.pdf) Accessed December 6, 2019.
- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. ["GloVe: Global Vectors for Word Representation."](https://nlp.stanford.edu/pubs/glove.pdf) Accessed February 1, 2020.
- T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. ["Advances in Pre-Training Distributed Word Representations."](https://arxiv.org/abs/1712.09405) Accessed February 17, 2020.
- Vikas Raunak. ["Simple and Effective Dimensionality Reduction for
Word Embeddings."](https://arxiv.org/pdf/1708.03629.pdf) Accessed February 27, 2020.
- Vikas Raunak, Vaibhav Kumar, Vivek Gupta, Florian Metze. ["On Dimensional Linguistic Properties of the Word Embedding Space."](https://arxiv.org/pdf/1910.02211.pdf) Accessed February 27, 2020.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. ["Language Models are Unsupervised Multitask Learners."](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Accessed February 27, 2020.
