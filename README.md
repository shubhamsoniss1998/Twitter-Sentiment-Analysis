# Twitter-Sentiment-Analysis
Introduction--

Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics. Therefore we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them. In this article, we aim to analyze the sentiment of the tweets provided from the Sentiment140 dataset by developing a machine learning pipeline involving the use of three classifiers (Logistic Regression, Bernoulli Naive Bayes, and SVM)along with using Term Frequency- Inverse Document Frequency (TFIDF). The performance of these classifiers is then evaluated using accuracy and F1 Scores.

Problem Statement--

In this project, we try to implement a Twitter sentiment analysis model that helps to overcome the challenges of identifying the sentiments of the tweets. The necessary details regarding the dataset are:
The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are:

target: the polarity of the tweet (positive or negative)

ids: Unique id of the tweet 

date: the date of the tweet 

flag: It refers to the query. 

If no such query exists then it is NO QUERY. 

user: It refers to the name of the user that tweeted
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



**ABSTRACT **

Twitter is a platform widely used by people to express their opinions and display sentiments on different occasions. Sentiment analysis is an approach to analyze data and retrieve sentiment that it embodies. Twitter sentiment analysis is an application of sentiment analysis on data from Twitter (tweets), in order to extract sentiments conveyed by the user. In the past decades, the research in this field has consistently grown. The reason behind this is the challenging format of the tweets which makes the processing difficult. The tweet format is very small which generates a whole new dimension of problems like use of slang, abbreviations etc. In this paper, we aim to review some papers regarding research in sentiment analysis on Twitter, describing the methodologies adopted and models applied, along with describing a generalized Python based approach.   

Keywords Sentiment analysis, Machine Learning, Natural Language Processing, Python.


**1. INTRODUCTION** 

Twitter has emerged as a major micro-blogging website, having over 100 million users generating over 500 million tweets every day. With such large audience, Twitter has consistently attracted users to convey their opinions and perspective about any issue, brand, company or any other topic of interest. Due to this reason, Twitter is used as an informative source by many organizations, institutions and companies. On Twitter, users are allowed to share their opinions in the form of tweets, using only 140 characters. This leads to people compacting their statements by using slang, abbreviations, emoticons, short forms etc. Along with this, people convey their opinions by using sarcasm and polysemy.  Hence it is justified to term the Twitter language as unstructured. In order to extract sentiment from tweets, sentiment analysis is used. The results from this can be used in many areas like analyzing and monitoring changes of sentiment with an event, sentiments regarding a particular brand or release of a particular product, analyzing public view of government policies etc. A lot of research has been done on Twitter data in order to classify the tweets and analyze the results. In this paper we aim to review of some researches in this domain and study how to perform sentiment analysis on Twitter data using Python. The scope of this paper is limited to that of the machine learning models and we show the comparison of efficiencies of these models with one another. 


**2. ABOUT SENTIMENT ANALYSIS **

Sentiment analysis is a process of deriving sentiment of a particular statement or sentence. It’s a classification technique which derives opinion from the tweets and formulates a sentiment and on the basis of which, sentiment classification is performed.  Sentiments are subjective to the topic of interest. We are required to formulate that what kind of features will decide for the sentiment it embodies. In the programming model, sentiment we refer to, is class of entities that the person performing sentiment analysis wants to find in the tweets. The dimension of the sentiment class is crucial factor in deciding the efficiency of the model. For example, we can have two-class tweet sentiment classification (positive and negative) or three class tweet sentiment classification (positive, negative and neutral). Sentiment analysis approaches can be broadly categorized in two classes – lexicon based and machine learning based. Lexicon based approach is unsupervised as it proposes to perform analysis using lexicons and a scoring method to evaluate opinions. Whereas machine learning approach involves use of feature extraction and training the model using feature set and some dataset. The basic steps for performing sentiment analysis includes data collection, pre-processing of data, feature extraction, selecting baseline features, sentiment detection and performing classification either using simple computation or else machine learning approaches. 


**2.1 Twitter Sentiment Analysis **

The aim while performing sentiment analysis on tweets is basically to classify the tweets in different sentiment classes accurately. In this field of research, various approaches have evolved, which propose methods to train a model and then test it to check its efficiency. Performing sentiment analysis is challenging on Twitter data, as we mentioned earlier. Here we define the reasons for this:  Limited tweet size: with just 140 characters in hand, compact statements are generated, which results sparse set of features.  Use of slang: these words are different from English words and it can make an approach outdated because of the evolutionary use of slangs.  Twitter features: it allows the use of hashtags, user reference and URLs. These require different processing than other words.  User variety: the users express their opinions in a variety of ways, some using different language in between, while others using repeated words or symbols to convey an emotion. 
All these problems are required to be faced in the preprocessing section. Apart from these, we face problems in feature extraction with less features in hand and reducing the dimensionality of features. 


**3.1 Tweet Collection**

Tweet collection involves gathering relevant tweets about the particular area of interest. The tweets are collected using Twitter’s streaming API [1], [3], or any other mining tool (for example WEKA [2]), for the desired time period of analysis. The format of the retrieved text is converted as per convenience (for example JSON in case of [3], [5]).  The dataset collected is imperative for the efficiency of the model. The division of dataset into training and testing sets is also a deciding factor for the efficiency of the model. The training set is the main aspect upon which the results depends.


**3.2 Pre-processing of tweets**

The preprocessing of the data is a very important step as it decides the efficiency of the other steps down in line. It involves syntactical correction of the tweets as desired. The steps involved should aim for making the data more machine readable in order to reduce ambiguity in feature extraction. Below are a few steps used for pre-processing of tweets -  
 Removal of re-tweets.
 Converting upper case to lower case: In case we are using case sensitive analysis, we might take two occurrence of same words as different due to their sentence case. It important for an effective analysis not to provide such misgivings to the model.
 Stop word removal: Stop words that don’t affect the meaning of the tweet are removed (for example and, or, still etc.). [3] uses WEKA machine learning package for this purpose, which checks each word from the text against a dictionary ([3], [5]).
 Twitter feature removal: User names and URLs are not important from the perspective of future processing, hence their presence is futile. All usernames and URLs are converted to generic tags [3] or removed [5]. 
 Stemming: Replacing words with their roots, reducing different types of words with similar meanings [3]. This helps in reducing the dimensionality of the feature set.
 Special character and digit removal: Digits and special characters don’t convey any sentiment. Sometimes they are mixed with words, hence their removal can help in associating two words that were otherwise considered different.
 Creating a dictionary to remove unwanted words and punctuation marks from the text [5].
 Expansion of slangs and abbreviations [5].
 Spelling correction [5].  Generating a dictionary for words that are important [7] or for emoticons [2].
 Part of speech (POS) tagging: It assigns tag to each word in text and classifies a word to a specific category like noun, verb, adjective etc. POS taggers are efficient for explicit feature extraction. 



**3.4 Sentiment classifiers**

** Bayesian logistic regression: **

selects features and provides optimization for performing text categorization. It uses a Laplace prior to avoid overfitting and produces sparse predictive models for text data. The Logistic Regression estimation        has the parametric form:  
        
    
                        Where     a normalization function, λ is is a vector of weight parameters for feature set and       is a binary function that takes as input a feature and a class label. It is triggered when a certain feature exists and the sentiment is hypothesized in a certain way [3].  
                        
** Naïve Bayes:**

It is a probabilistic classifier with strong conditional independence assumption that is optimal for classifying classes with highly dependent features. Adherence to the sentiment classes is calculated using the Bayes theorem.                        X is a feature vector defined as X = {  ,   ….  } and    is a class label. Naïve Bayes is a very simple classifier with acceptable results but not as good as other classifiers.


 Support Vector Machine Algorithm:
Support vector machines are supervised models with associated learning algorithms that analyze data used for classification and regression analysis [6], [9]. It makes use of the concept of decision planes that define decision boundaries.                X is feature vector, ‘w’ is weights of vector and ‘b’ is bias vector.     is the non-linear mapping from input space to high dimensional feature space. SVMs can be used for pattern recognition [2].


 Artificial Neural Network:
the ANN model used for supervised learning is the Multi-Layer Perceptron, which is a feed forward model that maps data onto a set of pertinent outputs. Training data given to input layer is processed by hidden intermediate layers and the data goes to the output layers. The number of hidden layers is very important metric for the performance of the model. There are two steps of working of MLP NN- feed forward propagation, involving learning features from feed forward propagation algorithm and back propagation, for cost function [5], [10].  Zimbra et al [1] propose an approach to use Dynamic Architecture for Artificial Neural Network (DAN2) which is a machine learned model with sufficient sensitivity to mild expression in tweets. They target to analyze brand related sentiments where occurrences of mild sentences are frequent. DAN2 is different than the simple neural networks as the number of hidden layers is not fixed before using the model. As the input is given, accumulation 


**4.2 Natural Language Processing (NLTK)**

Natural Language toolkit (NLTK) is a library in python, which provides the base for text processing and classification. Operations such as tokenization, tagging, filtering, text manipulation can be performed with the use of NLTK.  The NLTK library also embodies various trainable classifiers (example – Naïve Bayes Classifier). NLTK library is used for creating a bag-of words model, which is a type of unigram model for text. In this model, the number of occurrences of each word is counted. The data acquired can be used for training classifier models. The sentiment of the entire tweets is computed by assigning subjectivity score to each word using a sentiment lexicon. 


**4.8 Feature Extraction**

Various methodologies for extracting features are available in the present day. Term frequency-Inverse Document frequency is an efficient approach. TF-IDF is a numerical statistic that reflects the value of a word for the whole document (here, tweet). Scikit-learn provides vectorizers that translate input documents into vectors of features. We can use library function TfidfVectorizer(), using which we can provide parameters for the kind of features we want to keep by mentioning the minimum frequency of acceptable features. 4
