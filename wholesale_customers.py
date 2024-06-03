# Part 2: Cluster Analysis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(columns=['Channel', 'Region'])
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	summary_stats = df.describe().T[['mean', 'std', 'min', 'max']]
	return summary_stats

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	standardized_df = (df - df.mean()) / df.std()
	return standardized_df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, init='random', n_init=1)
    return pd.Series(kmeans.fit_predict(df))

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    return pd.Series(kmeans.fit_predict(df))


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    clustering = AgglomerativeClustering(n_clusters=k).fit_predict(df)
    return pd.Series(clustering)

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
    return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    results = []
    k_values = [3, 5, 10]

    # Original data
    for k in k_values:
        # KMeans
        for i in range(10):
            kmeans_clusters = kmeans(df, k)
            kmeans_score = clustering_score(df, kmeans_clusters)
            results.append({'Algorithm': 'KMeans', 'data': 'Original', 'k': k, 'Silhouette Score': kmeans_score})

        # Agglomerative
        agglo_clusters = agglomerative(df, k)
        agglo_score = clustering_score(df, agglo_clusters)
        results.append({'Algorithm': 'Agglomerative', 'data': 'Original', 'k': k, 'Silhouette Score': agglo_score})

    # Standardized data
    scaled_data = standardize(df)

    for k in k_values:
        # KMeans with standardized data
        for i in range(10):
            kmeans_clusters = kmeans(scaled_data, k)
            kmeans_score = clustering_score(scaled_data, kmeans_clusters)
            results.append({'Algorithm': 'KMeans', 'data': 'Standardized', 'k': k, 'Silhouette Score': kmeans_score})

        # Agglomerative with standardized data
        agglo_clusters = agglomerative(scaled_data, k)
        agglo_score = clustering_score(scaled_data, agglo_clusters)
        results.append({'Algorithm': 'Agglomerative', 'data': 'Standardized', 'k': k, 'Silhouette Score': agglo_score})

    return pd.DataFrame(results)

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    best_score = rdf['Silhouette Score'].max()
    return best_score

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):

    scaled_data = standardize(df)

    clusters = kmeans(scaled_data, 3)

    n_attributes = df.shape[1]
    attribute_names = df.columns

    plt.figure(figsize=(15, 10))
    for i in range(n_attributes):
        for j in range(i + 1, n_attributes):
            plt.subplot(n_attributes, n_attributes, i * n_attributes + j + 1)
            plt.scatter(df.iloc[:, i], df.iloc[:, j], c=clusters, cmap='viridis', alpha=0.5)
            plt.xlabel(attribute_names[i])
            plt.ylabel(attribute_names[j])

    plt.tight_layout()
    plt.savefig('plots.png')

