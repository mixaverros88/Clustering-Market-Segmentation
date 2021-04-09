########################################################################## START ############################################
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
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
from functions import *
import time
import os

font = {'family' : 'DejaVu Sans','weight' : 'normal','size' : 6} # decrease font size due to diplay issues of many columns
plt.rc('font', **font)
elapsed_time = {"kmeans": [],"gmm": [] ,"hierarchy": [],"dbscan": [] } # Copute the computational time of every algorith
missing_values = ['n/a', 'na', '--', '?'] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990.data.txt', sep=',', nrows=20000, na_values=missing_values)
print('Initial shape: ', df.shape) 
        
# df=df.sample(frac=0.2, replace=True, random_state=1)
displayBoxPlots(df, ['caseid','iRemplpar'])  # display box plots for every column

############################################################# Data Preprocessing  ############################################

groupedByUserId  = df.groupby(['caseid']) # group rows per user_id
df.insert(0, 'military', 0 , True) # create a column 
for index,group in groupedByUserId: # loop through every row and if a user a leas has took place in a war add in to military column 1 or 0
    if group['iFeb55'].values[0] == 1 or group['iKorean'].values[0] == 1 or group['iMay75880'].values[0] == 1 or group['iRvetserv'].values[0] == 1 or group['iSept80'].values[0] == 1 or group['iVietnam'].values[0] == 1 or group['iWWII'].values[0] == 1 or group['iOthrserv'].values[0] == 1:
        df.loc[ df['caseid']== index, 'military'] = 1
    else:
        df.loc[ df['caseid']== index, 'military'] = 0

# Missing values
print('Missing values: ', df.isnull().values.sum())

df.drop('caseid', axis=1, inplace=True) 
print('Count all the unique values before pre-processing: ', df.nunique().sum())

# Drop Military related columns 
df.drop('iFeb55', axis=1, inplace=True) 
df.drop('iKorean', axis=1, inplace=True)
df.drop('iMay75880', axis=1, inplace=True) 
df.drop('iRvetserv', axis=1, inplace=True)
df.drop('iSept80', axis=1, inplace=True)
df.drop('iVietnam', axis=1, inplace=True)
df.drop('iWWII', axis=1, inplace=True)
df.drop('iOthrserv', axis=1, inplace=True)
df.drop('iSubfam1', axis=1, inplace=True) # drop both iSubfam1, iSubfam2 since in iSubfam2 the most of the cases are 0 
df.drop('iSubfam2', axis=1, inplace=True)
df.drop('dHispanic', axis=1, inplace=True) # since tha most cases are not hispanic
df.drop('iRelat2', axis=1, inplace=True) # since tha most cases are N/a Gq/not Other Rel.

# iRemplpar
# 0: Both Parents Works
# 1: Only Father Works
# 2: Only Mather Works
# 3: Neither Parent Works
df.loc[df['iRemplpar'] == 111] = 0  # Both Parents Works
df.loc[(df['iRemplpar'] == 112) | (df['iRemplpar'] == 121) | (df['iRemplpar'] == 122) | (df['iRemplpar'] == 211) | (df['iRemplpar'] == 212) | (df['iRemplpar'] == 213) , 'iRemplpar'] = 1 # Only Father Works
df.loc[(df['iRemplpar'] == 113) | (df['iRemplpar'] == 133) | (df['iRemplpar'] == 134) | (df['iRemplpar'] == 221) | (df['iRemplpar'] == 222) | (df['iRemplpar'] == 223) , 'iRemplpar'] = 2 # Only Mather Works
df.loc[(df['iRemplpar'] == 114) | (df['iRemplpar'] == 141) , 'iRemplpar'] = 3 # Neither Parent Works

# iRelat1
# Relative : 0
# No Relative : 1
df.loc[(df['iRelat1'] >= 0) & (df['iRelat1'] <= 6) , 'iRelat1'] = 0 # Relative
df.loc[(df['iRelat1'] >= 7) & (df['iRelat1'] <= 13) , 'iRelat1'] = 1 # No Relative

# iRiders
# 0: N/a Not a Worker or Worker Whose Means o
# 1: Drove Alone
# 2: More than 2 people
df.loc[(df['iRiders'] >= 2) & (df['iRiders'] <= 8) , 'iRiders'] = 2 # more than 2 people

# iTmpabsnt
# 0: N/a Less Than 16 Yrs. Old/at Work/did No
# 1: Yes
# 2: No
df.loc[(df['iTmpabsnt'] == 1) | (df['iTmpabsnt'] == 2) , 'iTmpabsnt'] = 1
df.loc[(df['iTmpabsnt'] == 3) , 'iTmpabsnt'] = 2

# iMilitary
# 0: No Military service
# 1: Military Service
df.loc[(df['iMilitary'] == 0) | (df['iMilitary'] == 4) , 'iMilitary'] = 0
df.loc[(df['iMilitary'] >= 1) & (df['iMilitary'] <= 3) , 'iMilitary'] = 1

# iLang1
# 0: No 
# 1: Yes
df.loc[(df['iLang1'] == 0) | (df['iLang1'] == 2) , 'iLang1'] = 0

# iMobility
# 0: No 
# 1: Yes
df.loc[(df['iMobility'] == 0) | (df['iMobility'] == 2) , 'iMobility'] = 0

# iMobillim
# 0: No 
# 1: Yes
df.loc[(df['iMobillim'] == 0) | (df['iMobillim'] == 2) , 'iMobillim'] = 0

# iFertil
# 0: No 
# 1: 2-4
# 2: having many children 5-13
df.loc[(df['iFertil'] == 0) | (df['iFertil'] == 1) , 'iFertil'] = 0
df.loc[(df['iFertil'] >= 2) & (df['iFertil'] <= 4) , 'iFertil'] = 1
df.loc[(df['iFertil'] >= 5) & (df['iFertil'] <= 13) ,'iFertil'] = 2

# iRspouse
# 0: No 
# 1: Yes
df.loc[(df['iRspouse'] == 0) | (df['iRspouse'] == 6) , 'iRspouse'] = 0
df.loc[(df['iRspouse'] >= 1) & (df['iRspouse'] <= 5) , 'iRspouse'] = 1

# iPerscare
# 0: No 
# 1: Yes
df.loc[df['iPerscare'] == 2, 'iPerscare'] = 0

# dRearning
# 0: No 
# 1: Medium Earning
# 2: Rich
df.loc[(df['dRearning'] >= 1) & (df['dRearning'] <= 3) , 'dRearning'] = 1
df.loc[(df['dRearning'] >= 4) & (df['dRearning'] <= 5) , 'dRearning'] = 2

# dPwgt1
# 0: Slim 
# 1: Normal
# 2: Obese
df.loc[(df['dPwgt1'] == 2) | (df['dPwgt1'] == 3) , 'dPwgt1'] = 2

# iMeans
# 0: Not
# 1: By own
# 2: Public Transportation
# 3: Other
df.loc[(df['iMeans'] == 1) | ((df['iMeans'] >= 7) & (df['iMeans'] <= 10)) , 'iMeans'] = 1
df.loc[(df['iMeans'] >= 2) & (df['iMeans'] <= 6) , 'iMeans'] = 2
df.loc[(df['iMeans'] == 11) | (df['iMeans'] == 12) , 'iMeans'] = 3

# iLooking
# 0: No 
# 1: Yes
df.loc[(df['iLooking'] == 0) | (df['iLooking'] == 2) , 'iLooking'] = 0
df.loc[ df['iLooking'] == 1, 'iLooking'] = 1

# iClass
# 0: No 
# 1: Yes
df.loc[(df['iClass'] == 0) | (df['iClass'] == 9) , 'iClass'] = 0
df.loc[(df['iClass'] != 0) & (df['iClass'] != 9) , 'iClass'] = 1

# iAvail
# 0: No 
# 1: Yes
df.loc[(df['iAvail'] >= 0) & (df['iAvail'] <= 3) , 'iAvail'] = 0
df.loc[ df['iAvail'] == 4, 'iAvail'] = 1

# iSchool
# 0: Not attend
# 1: Attend
df.loc[ df['iSchool'] == 1, 'iSchool'] = 0
df.loc[(df['iSchool'] >= 2) & (df['iSchool'] <= 3) , 'iSchool'] = 1

# iImmigr
# 0: Came to US before 1950
# 1: Came to US after 1950
df.loc[(df['iImmigr'] >= 1) & (df['iImmigr'] <= 9) , 'iImmigr'] = 0
df.loc[df['iImmigr'] == 10, 'iImmigr'] = 1

# iMarital
# 0: Never Married
# 1: Married 
df.loc[(df['iMarital'] >= 0) & (df['iMarital'] <= 3) , 'iMarital'] = 1
df.loc[df['iMarital'] == 4, 'iMarital'] = 0

# iYearsch
# 0: No School Completed
# 1: Median Education
# 3: High Education
df.loc[(df['iYearsch'] == 0) | (df['iYearsch'] == 1) , 'iYearsch'] = 0
df.loc[(df['iYearsch'] >= 2) & (df['iYearsch'] <= 10) , 'iYearsch'] = 1
df.loc[(df['iYearsch'] >= 10) & (df['iYearsch'] <= 17) , 'iYearsch'] = 2

# iEnglish      
# 0: Not Speak English
# 1: Speak English
df.loc[(df['iEnglish'] == 4) , 'iEnglish'] = 0
df.loc[(df['iEnglish'] >= 1) & (df['iEnglish'] <= 3) , 'iEnglish'] = 1

# iRagechld
# 0: No 
# 1: Yes
df.loc[(df['iRagechld'] == 0) | (df['iRagechld'] == 4) , 'iRagechld'] = 0
df.loc[(df['iRagechld'] >= 1) & (df['iRagechld'] <= 3) , 'iRagechld'] = 1

# dTravtime
# 0: No 
# 1: Below 1 hour
# 2: Above 1 hour
df.loc[(df['dTravtime'] >= 1) & (df['dTravtime'] <= 5) , 'dTravtime'] = 1
df.loc[(df['dTravtime'] == 6), 'dTravtime'] = 2

# iYearwrk
# 0: N/a Less Than 16 Yrs. Old
# 1: From 1980 until 1990
# 2: 1979 or Earlier
# 3: Never Worked
df.loc[(df['iYearwrk'] >= 1) & (df['iYearwrk'] <= 5) , 'iYearwrk'] = 1
df.loc[(df['iYearwrk'] == 6), 'iYearwrk'] = 2
df.loc[(df['iYearwrk'] == 7), 'iYearwrk'] = 3

# iCitizen
# 0: Born in US
# 1: Not Born in US
df.loc[(df['iCitizen'] == 4) , 'iCitizen'] = 0
df.loc[(df['iCitizen'] >= 1) & (df['iCitizen'] <= 3) , 'iCitizen'] = 1

# iRPOB
# 0: Citizen
# 1: Not a Citizen
df.loc[df['iRPOB'] != 52, 'iRPOB'] = 0
df.loc[df['iRPOB'] == 52, 'iRPOB'] = 1

print('Count all the unique values after pre-processing: ', df.nunique().sum())
displayBoxPlots(df) 

########################################################################## Transform Categorical Data ##########################################################################

print('Before Converting categorical variable into dummy/indicator variables: ', df.shape) 
transformedDF = df.copy()
for column in transformedDF:
    catLen = len(transformedDF[column].value_counts().index) # count the categorical values per column since is not need to convert the columns that has only 2 classes
    if catLen > 2:
        transformedDF = pd.get_dummies(transformedDF, columns=[column], prefix=[column] )

print('After Converting categorical variable into dummy/indicator variables: ', transformedDF.shape) 


########################################################################## Scale Data ##########################################################################

scaled_features = transformedDF.copy()
columns = scaled_features[transformedDF.columns]
std_scale = StandardScaler().fit(columns.values)
X = std_scale.transform(columns.values)

########################################################################## HOPKINS TEST ##########################################################################

# pyclustertend is a python package specialized in cluster tendency. Cluster tendency consist to assess if clustering algorithms are relevant for a dataset.
# https://github.com/lachhebo/pyclustertend
# https://www.kaggle.com/lachhebo/hopkins-test
# print('Hopkins Test:', hopkins(X, X.shape[0]) ) # Usually, we can believe in the existence of clusters when the hopkins score is bellow 0.25.

########################################################################## PCA ##########################################################################

pca = PCA()
pca.fit(X)
plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of PCA Components')
plt.ylabel('PCA Explained Variance Ratio %')
plt.title('PCA Variance Ration')
plt.show()

pca_cencus = PCA(n_components=10)
principalComponents_breast = pca_cencus.fit_transform(X)
principal_cencus_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10'])

# print('explained_variance_ratio_', pca_cencus.explained_variance_ratio_)

plt.scatter(principal_cencus_Df['principal component 1'],principal_cencus_Df['principal component 2'])
plt.title('PCA Distribution')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()

############################################################### Partitional clustering: KMEANS ##########################################################################

# plotSilhouette(X, scaled_features)
# plotElbowMethod(X)
# computeSilhouetteScore(X)

start_kmeans = time.time()
km = KMeans(n_clusters=3,init='k-means++',random_state=200)
kmeasPredictedLabels = km.fit_predict(X)
end_kmeans = time.time()

#km.cluster_centers_ # get cluster centers
#km.n_iter_# The number of iterations required to converge

print ('KMEANS Inertia Score: ',round(km.inertia_,2)) # The lowest SSE value
print ('KMEANS Silhouette Score: ', round(np.mean(metrics.silhouette_samples(X, kmeasPredictedLabels)),2)) # Silhouette: higher values are better

principal_cencus_Df['cluster_kmeans'] = kmeasPredictedLabels # add predicted clusters

# Visualize Clusters
df1 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==0]
df2 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==1]
df3 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==2]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green', label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red', label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='black', label='cluster 3')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('KMEANS Clustering')
plt.legend()
plt.show()

# Plot Parallel Coordinate plot for the Centroids
columnIndexes = []
columns = ["dAge_0","dAge_1","dAge_2","dAge_3","dAge_4","dAge_5","dAge_6","dAge_7", "dIncome1_0","dIncome1_1","dIncome1_2","dIncome1_3","dIncome1_4", "iSex", "iCitizen", "iMarital", "iFertil_0","iFertil_1","iFertil_2"]
for col in columns:
    columnIndexes.append(transformedDF.columns.get_loc(col))

# Create a data frame containing our centroids
centroids = pd.DataFrame(km.cluster_centers_[:, columnIndexes], columns=columns)
centroids['cluster'] = centroids.index
display_parallel_coordinates_centroids(centroids, 3)

principal_cencus_Df.drop('cluster_kmeans',axis=1,inplace=True) # delete unneeded column
############################################################## Density-based clustering: DBSCAN ##########################################################################

# calculateNearestNeighbors(X)
# computeEps(X,[14,15,16,17,18,19,20,21])
# computeMinSamples(X,[30,35,36,37,38,39,45,50])

start_dbscan = time.time()
dbscan = DBSCAN(eps=20, min_samples=36)
dbscanPredictedLabels = dbscan.fit_predict(X)
end_dbscan = time.time()

clusters = dbscan.labels_ # The labels_ property contains the list of clusters and their respective points.

n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0) # Number of clusters in labels, ignoring noise if present.
n_noise_ = list(clusters).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

silhouette_values = metrics.silhouette_samples(X, clusters)
print('DBSCAN Silhouette Score:', round(np.mean(silhouette_values),2)) # Silhouette: higher values are better

principal_cencus_Df['cluster_dbscan']=dbscanPredictedLabels # add predicted clusters

print('DBSCAN size per cluster',principal_cencus_Df.groupby(['cluster_dbscan']).size()) 

# Visualize Data
df1 = principal_cencus_Df [ principal_cencus_Df.cluster_dbscan== 0  ]
df2 = principal_cencus_Df [ principal_cencus_Df.cluster_dbscan== 1  ]
df3 = principal_cencus_Df [ principal_cencus_Df.cluster_dbscan== 2  ]
df4 = principal_cencus_Df [ principal_cencus_Df.cluster_dbscan== -1 ]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green', label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red', label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='yellow', label='cluster 3')
plt.scatter(df4['principal component 1'],df4['principal component 2'],color='black', label='Outliers')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()

principal_cencus_Df.drop('cluster_dbscan',axis=1,inplace=True) # delete unneeded column
#################################################### Hierarchical clustering:Agglomerative Hierarchical Clustering #######################################################

hiercluster = AgglomerativeClustering(affinity='euclidean', linkage='ward', compute_full_tree=True) # Create a hierarchical clustering model
# findNumberOfClusterInHierarchical(hiercluster,[2,3,4],X) # run the model for several clusters 

# Read off 8 clusters:
hiercluster.set_params(n_clusters=3)
clusters = hiercluster.fit_predict(X) 
print('Count of data points in each cluster\n', np.bincount(clusters)) # count of data points in each cluster

print ('Agglomerative Hierarchical Silhouette Score from 3 clusters : ', round(np.mean(metrics.silhouette_samples(X, clusters)),2)) # Silhouette: higher values are better

# Add cluster number to the original data
X_scaled_clustered = pd.DataFrame(X, columns=transformedDF.columns, index=transformedDF.index)
X_scaled_clustered['cluster'] = clusters

# Find the size of the clusters
print('Find the size of each cluster\n', X_scaled_clustered["cluster"].value_counts())

# Show a dendrogram, just for the first smallest cluster
from scipy.cluster.hierarchy import linkage, fcluster 
sample = X_scaled_clustered[X_scaled_clustered.cluster==0]
Z = linkage(sample, 'ward') 
names = sample.index 
plot_dendrogram(Z, names, figsize=(10,15))

# Generate An Image With the whole dendogram
link_matrix = linkage(sample, 'ward')
fig, ax = plt.subplots(1, 1, figsize=(20,200))
ax.set_title('Hierarchical Clustering Dendrogram')
ax.set_xlabel('distance')
ax.set_ylabel('name')
dendrogram(
    link_matrix,
    orientation='left',
    leaf_rotation=0., 
    leaf_font_size=12.,  
    labels=sample.index.values
)    
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
plt.savefig(r''+my_path+'\\images\\dendogram.png', dpi=320, format='png', bbox_inches='tight')

X_scaled_clustered2 = pd.DataFrame(transformedDF.loc[columnIndexes], columns=columns, index=transformedDF.index)
X_scaled_clustered2['cluster'] = clusters

# Plot Parallel Coordinate plot for the Centroids
means =  X_scaled_clustered2.groupby(by="cluster").mean()
display_parallel_coordinates_centroids(means.reset_index(), 3)

start_hierarchy = time.time()
hierarchyPredictedLabels = hiercluster.fit_predict(X)
end_hierarchy = time.time()

print ('Agglomerative Hierarchical Silhouette Score: ', round(np.mean(metrics.silhouette_samples(X, hierarchyPredictedLabels)),2)) # Silhouette: higher values are better

principal_cencus_Df['cluster_hierarchy'] = hierarchyPredictedLabels

# Visualise Clusters
df1 = principal_cencus_Df [ principal_cencus_Df.cluster_hierarchy == 0  ]
df2 = principal_cencus_Df [ principal_cencus_Df.cluster_hierarchy == 1  ]
df3 = principal_cencus_Df [ principal_cencus_Df.cluster_hierarchy == 2  ]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green',  label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red',    label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='yellow', label='cluster 3')
plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()

principal_cencus_Df.drop('cluster_hierarchy',axis=1,inplace=True) # delete unneeded column
########################################################################## Gaussian Mixture Model ############################################################

start_gmm = time.time()
gmm = GaussianMixture()
#findNumberOfClusterInGaussianMixture(gmm,[2,3,4,5,6,7,8,9],X)
gmm = GaussianMixture(n_components=3)
labels = gmm.fit_predict(X)
end_gmm = time.time()
principal_cencus_Df['cluster_gaussian']=labels # add predicted clusters

print ("Gaussian Mixture Silhouette Score: ", round(metrics.silhouette_score(X, labels),3)) # Silhouette: higher values are better

# Visualise Clusters
df1 = principal_cencus_Df [ principal_cencus_Df.cluster_gaussian== 0  ]
df2 = principal_cencus_Df [ principal_cencus_Df.cluster_gaussian== 1  ]
df3 = principal_cencus_Df [ principal_cencus_Df.cluster_gaussian== 2  ]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green',  label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red',    label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='yellow', label='cluster 3')
plt.title('Gaussian Mixture Model')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()

principal_cencus_Df.drop('cluster_gaussian',axis=1,inplace=True) # delete unneeded column

# TODO: plot likelihood
# Evaluate Model Performance — Mean Silhouette Coefficient
# https://medium.com/@tarammullin/dbscan-2788cfce9389

elapsed_time["kmeans"].append(round(end_kmeans-start_kmeans,2))
elapsed_time["gmm"].append(round(end_gmm-start_gmm,2))
elapsed_time["hierarchy"].append(round(end_hierarchy-start_hierarchy,2))
elapsed_time["dbscan"].append(round(end_dbscan-start_dbscan,2))

for x in elapsed_time:
  print('Computation Time of ' + x + ':', elapsed_time[x])
########################################################################## END ##########################################################################