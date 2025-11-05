# Advanced Topics in Analytics

*Instructor: Sergio A. Mora Pardo*

- email: <sergioa.mora@javeriana.edu.co>
- github: [sergiomorapardo](https://github.com/sergiomorapardo)


Knowledge of the challenges and solutions present in specific situations of organizations that require advanced and special handling of information, such as text mining, process mining, data flow mining (stream data mining) and social network analysis. This module on Natural Language Processing  will explain how to build systems that learn and adapt using real-world applications. Some of the topics to be covered include text preprocessing, text representation, modeling of common NLP problems such as sentiment analysis, similarity, recurrent models, word embeddings, introduction to lenguage generative models. The course will be project-oriented, with emphasis placed on writing software implementations of learning algorithms applied to real-world problems, in particular, language processing, sentiment detection, among others.


## Requiriments 
* [Python](http://www.python.org) version >= 3.7;
* [Numpy](http://www.numpy.org), the core numerical extensions for linear algebra and multidimensional arrays;
* [Scipy](http://www.scipy.org), additional libraries for scientific programming;
* [Matplotlib](http://matplotlib.sf.net), excellent plotting and graphing libraries;
* [IPython](http://ipython.org), with the additional libraries required for the notebook interface.
* [Pandas](http://pandas.pydata.org/), Python version of R dataframe
* [Seaborn](https://seaborn.pydata.org/), used mainly for plot styling
* [scikit-learn](http://scikit-learn.org), Machine learning library!

A good, easy to install option that supports Mac, Windows, and Linux, and that has all of these packages (and much more) is the [Anaconda](https://www.anaconda.com/).

GIT!! Unfortunatelly out of the scope of this class, but please take a look at these [tutorials](https://help.github.com/articles/good-resources-for-learning-git-and-github/)

## Evaluation

* 50% Project
* 40% Exercises
* 10% Class participation

## Deadlines
| Session | Activity | Deadline | Comments |
| :---- | :----| :---- | :---- |
| Deep Learning | Exercises<br>Project| March 21th | Expo March 22th |
| NLP | Exercises<br>Project| April 25th<br>April 11th | Expo April 12th |
| Graph Learning | Exercises<br>Project| May 24th | |
| Final grade | project | May 31thth | |


## Slack Channel
[Join here! <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Slack_icon_2019.svg/2048px-Slack_icon_2019.svg.png" width="40" height="40" >](https://join.slack.com/t/advancedtopic-ifp1252/shared_invite/zt-31dvj2p0j-N8tjsMZkyZWC70iENzgy~g)

## Schedule

### Basic Methods MLOps
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| March 1st | Machine Learning Operations (MLOps) | [Intro MLOps](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/MLOps.pdf) | |
| March 1st | ML monitoring & Data Drift | [Intro Data Drift](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/IntroDataDrift.ipynb)<br>[L2 - Intro Data Drift](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/default_stattest_adult.ipynb)<br>[L3 - Intro Model Monitoring](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/ibm_hr_attrition_model_validation.ipynb) | [E1 - Data Drift in Used Vehicle Price Prediction](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E2-UsedVehiclePricePredictionDrift.ipynb) |
| March 1st | Machine Learning as a Service (AIaaS) | [1 - Intro to APIs](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/IntroductionToAPIs.ipynb)<br>[L1 - Model Deployment](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Model_Deployment.ipynb) | [E2 - Model Deployment in Used Vehicle Price Prediction](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E1-UsedVehiclePricePredictionDeployment.ipynb) |

### Intro Deep Learning
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| March 8th | First steps in deep learning | [3 - Intro Deep Learning](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Intro%20Deep%20Learning.pdf)<br>[L3 - Introduction Deep Learining MLP](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L11-IntroductionDeepLearningMLP.ipynb)<br>[L4 - Simple Neural Network (handcraft)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/SimpleNeuralNetwork.ipynb)<br>[L5 - Simple Neural Network (Images)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/SimpleNeuralNetworkImage.ipynb)<br>[L6 - Deep Learning with Keras](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/DeepLearningTensorflow.ipynb)<br>[L7 - Deep Learning with Pytorch](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/DeepLearningPyTorch.ipynb) | [E3 - Neural Networks in Keras and PyTorch](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksKeras.ipynb) |
| March 15th | Deep Computer Vision | [4 - Convolutional Neural Networks](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Deep%20Computer%20Vision.pdf)<br>[L5 - CNN with TensorFlow](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/CNN-TensorFlow.ipynb)<br>[L6 - CNN with PyTorch 🔥](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/CNN-PyTorch.ipynb)<br>[L7 - Tranfer Learning with TensorFlow](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/PretrainedModelsTensorFlow.ipynb) | [E4 - Tranfer Learning with PyTorch](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E4-PretrainedModelsPytorch.ipynb) |
| March 22th | Computer Vision Project | Exercises Deadline | [P1 - Frailejon Detection (a.k.a "Big Monks Detection")](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/P0_BigMonksDetection.ipynb) |

### Intro Natural Language Processing
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| March 22th | Introduction to NLP | [1 - Introduction to NLP](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Introduction%20to%20NLP.pdf)<br>[2 - NLP Pipeline](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/NLP%20Pipeline.pdf)<br>[E1 - Tokenization](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L1-Tokenization.ipynb) | |

### Text Representation
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| March 22th | Space Vector Models | [1 - Basic Vectorizarion Approaches](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Basic%20Vectorizarion%20Approaches.pdf)<br>[L2 - OneHot Encoding](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L1-OneHotEncoding.ipynb)<br>[L3 - Bag of Words](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L2-BagOfWords.ipynb)<br>[L4 - N-grams](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L3-ngrams.ipynb)<br>[L5 - TF-IDF](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L4-TFIDF.ipynb)<br>[L6 - Basic Vectorization Approaches](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L5-BasicVectorizationApproaches.ipynb) | [E2 - Sentiment Analysis](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E1-SentimentPrediction.ipynb) |
| March 29th | Distributed Representations | [2 - Word Embbedings](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Word%20Embeddings.pdf)<br>[L7 - Text Similarity](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L7-TextSimilarity.ipynb)<br>[L8 - Exploring Word Embeddings](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L8-ExploringWordEmbeddings.ipynb)<br>[L9 - Song Embeddings](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L9-SongEmbeddings.ipynb)<br>[L10 - Visualizing Embeddings](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L10-VisualizingEmbeddingsUsingTSNE.ipynb)| [E2 - Homework Analysis (Bonus)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E2-HomeworksAnalysis.ipynb)<br>[E3 - Song Embedding Visualization](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E3-SongEmbeddingsVisualization.ipynb)<br>[E4 - Spam Classification](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E4-SpamClassification.ipynb)|

### NLP with Deep Learning 
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| April 5th | Deep Learning in NLP (RNN, LSTM, GRU) | [4 - RNN, LSTM, GRU](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/RNN%2C%20LSTM%20and%20GRU.pdf)<br>L12 - NLP with Keras<br>[L11 - NLP with Keras](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L12-DeepLearning_keras_NLP.ipynb)<br>[L13 - Recurrent Neural Network and LSTM](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L13-RecurrentNeuralNetworks_LSTM.ipynb)<br>[L14 - Headline Generator](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L14-Headline_Generator.ipynb) | [E5 - Neural Networks in Keras for NLP](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksKerasNLP.ipynb)<br>[E6 - Neural Networks in PyTorch for NLP](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E5-NeuralNetworksPyTorchNLP.ipynb)<br>[E7 - RNN, LSTM, GRU](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E6-RNN_LSTM_GRU.ipynb)|
| April 12th | NLP Project | | [P1 - Movie Genre Prediction](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/P1-MovieGenrePrediction.ipynb) |
| April 12th | Attention, Tranformers and BERT | [5 - Encoder-Decoder](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Encoder-Decoder.pdf)<br>[6 - Attention Mechanisms and Transformers](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Attention%20Mechanism.pdf)<br>[7 - BERT and Family](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/BERT.pdf)<br>[L16 - Positional Encoding](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L15-transformer_positional_encoding_graph.ipynb)<br>[L17 - BERT for Sentiment Clasification](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L16-BERT_for_sentiment_classification.ipynb)<br>[L18 - Transformers Introduction](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L17-TransformersIntroduction.ipynb) | [E8 - Text Summary](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E7-TextSummary.ipynb)<br>[E9 - Question Answering](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E8-QuestionAnswer.ipynb)<ol><li>[E10 - Open AI](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E9-OpenAI.ipynb)</li></ol> |
| April 19th | _Holy Week_ | _Holy Week_ | _Holy Week_ |
| April 25th | | Exercises Deadline | |

### Intro Graph
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| April 26th | Intro to Graphs | [Intro to Graphs](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/introGRaphs.pdf)<br>[L19 - Intro to Graphs](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L18-IntroductionGraphs.ipynb) | |
| April 26th | Graphs Metrics | [L20 - Graph Metrics](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L19-GraphMetrics.ipynb)<br>[L21 - Graphs Benchmarks](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L20-GraphsBenchmarks.ipynb)<br>[L22 - Facebook Analysis](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L21-FacebookNetworkAnalysis.ipynb) | [E10 - Twitter Analysis](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E10-TwitterNetworkAnalysis.ipynb) |

### Graph Representation Learning
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| May 3th | Graph Representation | [Graph Representations](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/GraphRepresentation.pdf)<br>[L23 - Graph Embedding](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L22-GraphEmbedding.ipynb)<br>[L24 - Deep Walk](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L23-DeepWalk.ipynb)<br>[L25 - Node2Vec](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L24-Node2Vec.ipynb)<br>[L26 - Recommendation System with Node2Vec](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L25-Node2Vec-RecSys.ipynb) | [E11 - Patent Citation Network (Node2Vec with RecSys)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/E11-PatentCitationNetwork.ipynb) |

### Intro to Geometric Deep Learning
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| May 10th | Graph Neural Network | [L27 - Graph Neural Networks - Node Features](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L27-GraphNeuralNetworks-NodeFeatures.ipynb)<br>[L28 - Graph Neural Networks - Node2Vec](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L26-Node2Vec-Pytorch.ipynb)<br>[L29 - Graph Neural Networks - Adjacency Matrix](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L27-GraphNeuralNetworks-AdjacencyMatrix.ipynb)<br>[L31 - Graph Neural Networks - Graph Convolutional Networks (GCN)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L29-GraphConvolutionalNetworks-NodeClassification.ipynb)<br>[L33 - Graph Neural Networks - Graph Attention Networks (GAT)](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L32-GraphAttentionNetworks.ipynb)<br>[L34 - Graph Convolutional Networks - Node Regression](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L31-GraphConvolutionalNetworks-NodeRegression.ipynb) | [L30 - Graph Neural Networks - Facebook Page-Page dataset](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L28-GraphNeuralNetworks.ipynb)<br>[L32 - Graph Convolutional Networks - Facebook Page-Page dataset](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L30-GraphConvolutionalNetworks-NodeClassification.ipynb)<br>[L34 - Graph Attention Networks - Cite Seer](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L33-GraphAttentionNetworks-CiteSeer.ipynb) |
| May 17th | Graph Machine Learning Task [Optional] | [L35 - Graph AutoEncoder - Link Prediction](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L35-LinkPredictionGraphAutoencoder.ipynb)<br>[L36 - Graph Variational AutoEncoder - Link Prediction \[extra\]](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L37-LinkPredictionVariationalAutoEncoder.ipynb)<br>[L37 - Node2Vec - Link Classification](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L36-LabelClassificationNode2Vec.ipynb)<br>[L38 - Graph Isomorphism Network - Graph Classification](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/L34-GraphClassificationGraphIsomorphismNetwork.ipynb) | |
| May 24th | Geometric Deep Learning Project | Exercises Deadline | [P3 - Graph Machine Learning](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/exercises/P2-GraphMachineLearning.pdf) / [P3 - Graph Machine Learning [old < 2022]](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/Proyecto_GML.pdf) |

### Grades
| Date | Session | Notebooks/Presentations | Exercises |
| :----| :----| :------------- | :------------- |
| May 31th | Final Grades | | |

## Interest Links 🔗
Module | Topic | Material |
| :----| :----| :----|
| NLP | Word Embedding Projector |[Tensorflow Embeddings Projector](https://projector.tensorflow.org/)|
| NLP | Time Series with LSTM | [ARIMA-SARIMA-Prophet-LSTM](https://www.kaggle.com/code/sergiomora823/time-series-analysis-arima-sarima-prophet-lstm) |
| NLP | Stanford | [Natural Language Processing with Deep Learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/)
| GML | Stanford | [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)

## Extra Material
Module | Topic | Material |
| :----| :----| :----|
| NLP | Polarity | [Sentiment Analysis - Polarity](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/X1-SentimentAnalysisPolarity.ipynb) |
| NLP | Image & Text | [Image Captions](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/X2-image_captions.ipynb) |
| ML | Hyperparameter Tuning [WIP] | [Exhaustive Grid Search]()<br>[Randomized Parameter Optimization]()<br>[Automate Hyperparameter Search]() |
| NLP | Neural Style Transfer | [Style Transfer](https://github.com/sergiomorapardo/AdvancedTopicsAnalytics/blob/main/notebooks/X3-style_transfer.ipynb) |
