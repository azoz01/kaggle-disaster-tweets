# Kaggle disaster tweets
Project is aimed to create model which is able to predict, if tweet is about disaster. <br/>
Link to kaggle, where dataset comes from: https://www.kaggle.com/c/nlp-getting-started .
## Used methods
To prediction task, are used neural networds. However there is logistic regressin, which gave suprisingly good results.
Used methods:
  * Logistic regression 0.78 accuracy on test set, but is overfitted
  * MLP - 0.77 accuracy and overfitting
  * Vanilla RNN - 0.79 accuracy
  * LSTM - 0.78 accuracy
  * BERT - 0.81 accuracy

  Above results are best ones which I succeeded to reach.
 
 ## Preprocessing
 Used preprocessing contains:
  * to lower conversion
  * removal of mails, ulrs, twitter references, punctuation, numbers, stopwords
  * spelling correction.
  * lemmatization (also stemming is available but didn't give better results)
  
 After that the following techniques were used for embedding:
  * one hot for logistic regression and MLP
  * word2vec from spacy for RNN and LSTM
 
 ## File structure

    ├── data - raw data
    ├── model_utils - utilities for models
    ├── models - notebooks with models
    │   ├──trained - trained models saved in .pt format
    ├── nn_datasets - torch datasets
    |   ├──precalculated_datasets - precalculated time-consuming models like those with word2vec embedding
    ├── prepared - data after some preprocessing
    └── preprocessing - preprocessing utilities
