import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(['fnlwgt'], axis=1)
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.values

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().values.sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	return df.columns[df.isnull().any()].tolist()

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	bachelors_masters_df = df[df['education'].isin(['Bachelors', 'Masters'])]
	percentage = (len(bachelors_masters_df) / len(df)) * 100
	return round(percentage, 1)

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df_na = df.dropna()
	return df_na

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	df_dropped = df.drop(['class'], axis=1)
	df_onehot = pd.get_dummies(df_dropped, columns=['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
	return df_onehot

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	labels_df = df['class']
	label_encoder = LabelEncoder()
	labels_encoded = label_encoder.fit_transform(labels_df)
	return pd.Series(labels_encoded)

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	clf = DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	prediction = clf.predict(X_train)
	return pd.Series(prediction)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	misclassified = (y_pred != y_true).sum()
	error_rate = misclassified / len(y_true)
	return error_rate



