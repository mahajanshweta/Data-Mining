import pandas as pd
import numpy as np
import requests
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin-1')
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique().tolist()

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	return df['Sentiment'].value_counts().index[1]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	extremely_positive_tweets = df[df['Sentiment']=='Extremely Positive']
	date_counts = extremely_positive_tweets['TweetAt'].value_counts()
	date_with_most_tweets = date_counts.idxmax()
	return date_with_most_tweets

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: ' '.join(x.split()))


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	tdf = df['OriginalTweet'].apply(lambda x: x.split())
	return tdf

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	return tdf.str.len().sum()

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	return tdf.apply(set).explode().nunique()

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	word_counts = tdf.explode().value_counts()
	return word_counts.head(k).index.tolist()

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stop_words_url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
	response = requests.get(stop_words_url)
	stop_words = set(response.text.splitlines())
	return tdf.apply(lambda tweet: [word for word in tweet if word not in stop_words and len(word) > 2])

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemmer = PorterStemmer()
	return tdf.apply(lambda tweet: [stemmer.stem(word) for word in tweet])

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	vectorizer = CountVectorizer(ngram_range=(1, 3),max_df=3)
	X_counts = vectorizer.fit_transform(df['OriginalTweet'])
	classifier = MultinomialNB()
	classifier.fit(X_counts, df['Sentiment'])
	prediction = classifier.predict(X_counts)
	return prediction

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	correct_predictions = np.sum(y_pred == y_true)
	total_predictions = len(y_true)
	accuracy = correct_predictions / total_predictions
	return round(accuracy, 3)



