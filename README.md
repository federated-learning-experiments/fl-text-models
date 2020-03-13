# fl-text-models
Federated learning with text DNNs for DATA 591 at University of Washington.  See the `README.md` file in **`final_research_report`** for an overview of our experiment designs and results.

##### Data
The main dataset used for these experiments is hosted by Kaggle and made available through the [tff.simulation.datasets module]((https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)) in the [Tensorflow Federated API](https://github.com/tensorflow/federated).  Stack Overflow owns the data and has released the data under the [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

##### Environment
Experiments were conducted in a Python 3.7 conda environment with the packages in `requirements.txt` on both GPU and CPU VMs running Ubuntu 16.04.

##### Running Experiments
To conduct experiments with our code:

- Clone the repository and replicate our conda environment.
- Configure the `params.json` file to set a simulated client data sampling strategy, pretraining approach, and federated model architecture.
- Execute `federated_nwp.py` to train a federated text model on Stack Overflow for next word prediction according to the desired parameters.  This script applies our methods described in `final_research_report/README.md` and is based on work from the research section of the [Tensorflow Federated API](https://github.com/tensorflow/federated).
- Model weights, train and validation statistics, plots, and client sample metadata are automatically stored in the `experiment_runs` directory.  Run `experiment_runs/training_progress.py` to summarize model performance during or after training.
- See the `notebooks` directory for additional analysis, experiments, and examples of loading, testing, and comparing trained models.

##### References
This project draws mainly from the following research, but other sources are referenced throughout this repository, particularly code snippets.  Special thanks to Keith Rush and Peter Kairouz from Google for their guidance throughout the course of this project.  

-	[1] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. [“Communication-Efﬁcient Learning of Deep Networks."](https://arxiv.org/pdf/1602.05629.pdf) Accessed December 6, 2019.
- [2] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander. [“Towards Federated Learning at Scale: System Design.”](https://arxiv.org/pdf/1902.01046.pdf) Accessed December 6, 2019.
- [3] Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy, Francoise Beaufays Sean Augenstein, Hubert Eichner, Chloe Kiddon, Daniel Ramage. [“Federated Learning for Mobile Keyboard Prediction.”](https://arxiv.org/pdf/1811.03604.pdf) Accessed December 6, 2019.
- [4] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. ["GloVe: Global Vectors for Word Representation."](https://nlp.stanford.edu/pubs/glove.pdf) Accessed February 1, 2020.
- [5] Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov. ["Enriching Word Vectors with Subword Information."](https://arxiv.org/pdf/1607.04606.pdf) Accessed February 17, 2020.
- [6] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. ["Language Models are Unsupervised Multitask Learners."](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Accessed February 27, 2020.
- [7] Jiaqi Mu, Pramod Viswanath ["All-but-the-Top: Simple and Effective Postprocessing for Word Representations"](https://openreview.net/forum?id=HkuGJ3kCb).  Accessed March 05, 2020.
- [8] Vikas Raunak, Vivek Gupta, Florian Metze. ["Effective Dimensionality Reduction for Word Embeddings"](https://www.aclweb.org/anthology/W19-4328.pdf). Accessed March 05, 2020.
- [9] Vikas Raunak, Vaibhav Kumar, Vivek Gupta, Florian Metze. ["On Dimensional Linguistic Properties of the Word Embedding Space."](https://arxiv.org/pdf/1910.02211.pdf) Accessed February 27, 2020.
- [10] Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan. ["Adaptive Federated Optimization"](https://arxiv.org/pdf/2003.00295.pdf) Accessed March 6, 2020.
- [11] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Jamie Brew. ["HuggingFace's Transformers: State-of-the-art Natural Language Processing"](https://arxiv.org/abs/1910.03771).  Accessed March 05, 2020.

##### Contact
- [Arjun Singh\*](https://github.com/sinarj)
- [Joel Stremmel\*](https://github.com/jstremme)
