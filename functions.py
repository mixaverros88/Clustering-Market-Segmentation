########################################################################## START ##########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from pyclustertend import hopkins
import time
import os

palette = sns.color_palette('bright', 10)

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle('Parallel Coordinates Plot for the Clusters', fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):   
        plt.xticks(rotation=40) 
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)      
    plt.xticks(rotation=40) 
    plt.show()

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle('Parallel Coordinates plot for the Centroids', fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20) 
    plt.xticks(rotation=90)      
    plt.show()
      
def plotElbowMethod(X): 
    # Plot Elbow Method
    wcss=[]
    for i in range(1,13):
        kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=200)
        kmeans_pca.fit(X)
        wcss.append(kmeans_pca.inertia_) # The lowest SSE value

    plt.plot(range(1,13),wcss, marker='o', linestyle ='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

def plot_dendrogram(Z, names, figsize=(10,25)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = 'left',
    )
    plt.show()

def plotSilhouette(X,scaled_features):
    kmeans_kwargs = {
        'init': 'random',
        'n_init': 10,
        'max_iter': 300,
        'random_state': 42,
    }
     # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = metrics.silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette')
    plt.show()

def displayBoxPlots(df, *dropColums):
    vizualiseDF=df.copy()
    for column in dropColums:
        vizualiseDF.drop(column, axis=1, inplace=True) 
    vizualiseDF_columns_count = len(vizualiseDF.columns)/2
    if isinstance(vizualiseDF_columns_count, float):
        vizualiseDF_columns_count = vizualiseDF_columns_count-0.5

    vizualiseDF_columns_count = int(vizualiseDF_columns_count)
    for i in range(2):
        step = i + 1
        sns.boxplot(data=vizualiseDF.iloc[:, vizualiseDF_columns_count*(i):vizualiseDF_columns_count*(step)])
        plt.xticks(rotation=90)
        plt.show()

def calculateNearestNeighbors(X):
    # In order to calculate the distance from each point to its closest neighbor we are using the NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    print('Distances', distances)
    print('AVG Distances', np.average(distances))
    # The optimal value for epsilon will be found at the point of maximum curvature.
    plt.title('Calculate the distance between 2 points')
    plt.plot(distances)
    plt.show()

def computeMinSamples(X,min_samples):
    # Compute best min_sample value
    #min_samples = [30,35,36,37,38,39,45,50]
    for i in min_samples:
        #print('min_samples value is ' + str(i))
        db = DBSCAN(eps=10, min_samples=i).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # Ingoring the label '-1' as its for the outliers
        labels = set([label for label in db.labels_ if label >= 0])
        #print(set(labels))
        print('For min_samples value : ' + str(i), 'Total no. of clusters are ' + str(len(set(labels))))

def computeEps(X, range_eps):
    # Compute best epsilon value, Find the bigest silhouette score
    #range_eps = [14,15,16,17,18,19,20,21]
    for i in range_eps:
        #print('eps value is ' + str(i))
        db = DBSCAN(eps=i, min_samples=45).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(set(labels))
        silhouette_avg = metrics.silhouette_score(X, labels)
        print('For eps value :' + str(i), labels, 'The average silhouette_score is : ', silhouette_avg)
        
def findNumberOfClusterInHierarchical(hiercluster,clusterNumber,X):
    for number in clusterNumber:
        hiercluster.set_params(n_clusters=number)
        clusters = hiercluster.fit_predict(X) 
        print('Count of data points in each cluster for ' + str(number) + ' Clusters : ', np.bincount(clusters)) # count of data points in each cluster     
        print ('Agglomerative Hierarchical Silhouette Score for ' + str(number) + ' Clusters : ', round(np.mean(metrics.silhouette_samples(X, clusters)),3)) # Silhouette: higher values are better

def findNumberOfClusterInGaussianMixture(gmm,clusterNumber,X):
    for number in clusterNumber:
        gmm.set_params(n_components=number)
        clusters = gmm.fit_predict(X) 
        print('Count of data points in each cluster for ' + str(number) + ' Clusters : ', np.bincount(clusters)) # count of data points in each cluster     
        print ('GaussianMixture Silhouette Score for ' + str(number) + ' Clusters : ', round(np.mean(metrics.silhouette_samples(X, clusters)),3)) # Silhouette: higher values are better

def computeSilhouetteScore(X):
    # Selecting optimal number of clusters in KMeans
    for i in range(2,10):
        kMeans_labels=KMeans(n_clusters=i,init='k-means++',random_state=200).fit(X).labels_
        print('Silhouette score for k(clusters): '+str(i)+' is '+str(metrics.silhouette_score(X,kMeans_labels,metric='euclidean',sample_size=1000,random_state=200)))


########################################################################## END ##########################################################################