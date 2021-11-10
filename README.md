# Twitter-Sentiment-Analysis
Introduction--
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics. Therefore we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them. In this article, we aim to analyze the sentiment of the tweets provided from the Sentiment140 dataset by developing a machine learning pipeline involving the use of three classifiers (Logistic Regression, Bernoulli Naive Bayes, and SVM)along with using Term Frequency- Inverse Document Frequency (TFIDF). The performance of these classifiers is then evaluated using accuracy and F1 Scores.
Problem Statement--
In this project, we try to implement a Twitter sentiment analysis model that helps to overcome the challenges of identifying the sentiments of the tweets. The necessary details regarding the dataset are:
The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are:
target: the polarity of the tweet (positive or negative) ids: Unique id of the tweet date: the date of the tweet flag: It refers to the query. If no such query exists then it is NO QUERY. user: It refers to the name of the user that tweeted
CLASSIFICATION DATA EXPLORATION MACHINE LEARNING PROJECT
text: It refers to the text of the tweet
Project Pipeline--
The various steps involved in the Machine Learning Pipeline are :
Import Necessary Dependencies
Read and Load the Dataset
Exploratory Data Analysis
Data Visualization of Target Variables
Data Preprocessing
Splitting our data into Train and Test Subset
Transforming Dataset using TF-IDF Vectorizer
Function for Model Evaluation
Model Building
Conclusion

