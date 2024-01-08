# Fake News Classification

The project uses the Fake and Real News Dataset downloaded from the Kaggle (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). The dataset includes two csv files - one contains texts of news articles labeled as 'fake' (23481 articles), another contains texts of news articles labeled as 'true' (21417 articles). Tables in the files contain information about title, text, subject and date when the article was uploaded.

The purpose of the project is to train machine learning and deep learning models which could be used in predicting whether a particular article is fake or not.

The project consists of two parts:

- the exploratory analysis of the dataset news articles and
 - building, training, fine-tuning and evaluating classifiers able to predict whether a particular article is fake or not.

The project includes these files:

- the file "news_analysis_with_corrections.ipynb" with the report on exploratory analysis, training, assessing and interpreting machine learning models and two linear layers deep learning model;
- the file "deep_learing_distilbert_1.ipynb" with the report on training, fine-tuning and assessment of a deep learning model based on the pre-trained transformer model - Distilbert with parameters:
  - learning rate 0.001 (training) 0.0001 (fine-tuning),
  - maximum sequence length was set to 300;
  - training-validation-test split 0.4x0.4x0.2;
- the file "deep_learing_distilbert_2.ipynb" the report on training, fine-tuning and assessment of a deep learning model based on the pre-trained transformer model Distilbert with parameters:
  - learning rate 0.0001 (training) 0.00001 (fine-tuning);
  - maximum sequence length was set to 128;
     training-validation-test split 0.7x0.15x0.15;
- the file "fake_news_dataset_distilbert.py" which contains classes for datasets' transformations.

