########################################################################## START ##########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import time
import os

elapsed_time = {"kmeans": [],"gmm": [] ,"hierarchy": [],"dbscan": [] } # Copute the computational time of every algorith
missing_values = ['n/a', 'na', '--', '?'] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990.data.txt', sep=',', nrows=2000, na_values=missing_values)
print('initial shape: ', df.shape) 

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

def computeMinSamples(X):
    # Compute best min_sample value
    min_samples = [10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26]
    for i in min_samples:
        #print('min_samples value is ' + str(i))
        db = DBSCAN(eps=18, min_samples=i).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        # Ingoring the label '-1' as its for the outliers
        labels = set([label for label in db.labels_ if label >= 0])
        #print(set(labels))
        print('For min_samples value = ' + str(i), 'Total no. of clusters are ' + str(len(set(labels))))

def computeEps(X):
    # Compute best epsilon value, Find the bigest silhouette score
    range_eps = [16,17,18,19]
    for i in range_eps:
        #print('eps value is ' + str(i))
        db = DBSCAN(eps=i, min_samples=15).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(set(labels))
        silhouette_avg = metrics.silhouette_score(X, labels)
        print('For eps value=' + str(i), labels, 'The average silhouette_score is : ', silhouette_avg)
        
# Data Preprocessing 
groupedByUserId  = df.groupby(['caseid']) # group rows per user_id
df.insert(0, 'military', 0 , True)
for index,group in groupedByUserId:
    if group['iFeb55'].values[0] == 1 or group['iKorean'].values[0] == 1 or group['iMay75880'].values[0] == 1 or group['iRvetserv'].values[0] == 1 or group['iSept80'].values[0] == 1 or group['iVietnam'].values[0] == 1 or group['iWWII'].values[0] == 1 or group['iOthrserv'].values[0] == 1:
        df.loc[ df['caseid']== index, 'military'] = 1
    else:
        df.loc[ df['caseid']== index, 'military'] = 0

# Print missing values
print('Print missing values', df.isnull().values.sum())

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

df.drop('caseid', axis=1, inplace=True) 
df.drop('dHispanic', axis=1, inplace=True) # since tha most cases are not hispanic
df.drop('iRelat2', axis=1, inplace=True) # since tha most cases are N/a Gq/not Other Rel.

# df.drop(index=df[df['dPoverty'] == 0].index,    inplace=True) # drop the rows that has N/A
# df.drop('iRemplpar', axis=1, inplace=True) # since iRemplpar column has many zero values
# df.drop(index=df[df['iSchool']  == 0].index,     inplace=True)   # N/a Less Than 3 Yrs. Old
# df.drop(index=df[df['iEnglish'] == 0].index,    inplace=True)    # N/a Less Than 5 Yrs. Old/speaks Only Eng
# df.drop(index=df[df['iImmigr'] == 0].index,    inplace=True)     # Born in the U.S. since we can take this value from the citizen column

#TODO
#   dDepart 
#   dYrsserv vs iMilitary
#   iRiders
#   YEARWRK ==> Never Worked
#   dAncsty1, dAncsty2, dDepart, iDisabl1, iDisabl2, dHour89, dHours, dIndustry, dOccup, iRalat1, iRlabor, dRpincome
#TODO


# REMPLPAR
# Employment Stat. of Parents
# 000: N/a Not Own Child of Hshldr., and Not Ch
# 111: Both Parents At Work 35 or More Hrs.
# 112: Father Only At Work 35 or More Hrs.
# 113: Mother Only At Work 35 or More Hrs.
# 114: Neither Parent At Work 35 or More Hrs.
# 121: Father At Work 35 or More Hrs.
# 122: Father Not At Work 35 or More Hrs.
# 133: Mother At Work 35 or More Hrs.
# 134: Mother Not At Work 35 or More Hrs.
# 141: Neither Parent in Labor Force
# 211: Father At Work 35 or More Hrs.
# 212: Father Not At Work 35 or More Hrs.
# 213: Father Not in Labor Force
# 221: Mother At Work 35 or More Hrs.
# 222: Mother Not At Work 35 or More Hrs.
# 223: Mother Not in Labor Force
df.loc[df['iRemplpar'] == 111] = 0  # Both Parents Works
df.loc[(df['iRemplpar'] == 112) | (df['iRemplpar'] == 121) | (df['iRemplpar'] == 122) | (df['iRemplpar'] == 211) | (df['iRemplpar'] == 212) | (df['iRemplpar'] == 213) , 'iRemplpar'] = 1 # Only Father Works
df.loc[(df['iRemplpar'] == 113) | (df['iRemplpar'] == 133) | (df['iRemplpar'] == 134) | (df['iRemplpar'] == 221) | (df['iRemplpar'] == 222) | (df['iRemplpar'] == 223) , 'iRemplpar'] = 2 # Only Mather Works
df.loc[(df['iRemplpar'] == 114) | (df['iRemplpar'] == 141) , 'iRemplpar'] = 3 # Neither Parent Works

# RIDERS
# Vehicle Occupancy
# 0:N/a Not a Worker or Worker Whose Means o
# 1:Drove Alone
# 2:2 People
# 3:3 People
# 4:4 People
# 5:5 People
# 6:6 People
# 7:7 to 9 People
# 8:10 or More People
df.loc[(df['iRiders'] >= 2) & (df['iRiders'] <= 8) , 'iRiders'] = 2 # more than 2 people


# TMPABSNT
# Temp. Absence From Work
# 0: N/a Less Than 16 Yrs. Old/at Work/did No
# 1: Yes, on Layoff
# 2: Yes, on Vacation, Temp. Illness, Labor D
# 3: No
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
# 1: Public Transportation
# 2: By own
# 3: Other
df.loc[(df['iMeans'] >= 2) | (df['iMeans'] <= 6) , 'iMeans'] = 1
df.loc[(df['iMeans'] == 1) | (df['iMeans'] >= 7) | (df['iMeans'] <= 10) , 'iMeans'] = 2
df.loc[ df['iMeans'] == 11, 'iMeans'] = 3
df.loc[ df['iMeans'] == 12, 'iMeans'] = 4

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

# Visualize Raw Data
plt.scatter(df['dAge'], df['dIncome1'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age and income distribution')
plt.show()

# for column in df:
#     plt.boxplot(df[column])
#     plt.title(column)
#     plt.show()

# for column in df:
#     sns.countplot(data=df, x=df[column])
#     plt.title('Count the distribution of '+ column + ' Feature')
#     plt.show()

########################################################################## Transform Categorical Data ##########################################################################

transformedDF = df.copy()
for column in transformedDF:
    transformedDF = pd.get_dummies(transformedDF, columns=[column], prefix=[column + '_Type_is'] )

print('After Converting categorical variable into dummy/indicator variables: ', transformedDF.shape) 

# Scale Data
scaled_features = transformedDF.copy()
columns = scaled_features[transformedDF.columns]
std_scale = StandardScaler().fit(columns.values)
X = std_scale.transform(columns.values)

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

print('explained_variance_ratio_', pca_cencus.explained_variance_ratio_)

plt.scatter(principal_cencus_Df['principal component 1'],principal_cencus_Df['principal component 2'])
plt.title('PCA Distribution')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()

########################################################################## KMEANS ##########################################################################
# Plot Elbow Method
wcss=[]
for i in range(1,21):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=200)
    kmeans_pca.fit(X)
    wcss.append(kmeans_pca.inertia_)

plt.plot(range(1,21),wcss, marker='o', linestyle ='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Selecting optimal number of clusters in KMeans
for i in range(2,10):
    kMeans_labels=KMeans(n_clusters=i,init='k-means++',random_state=200).fit(X).labels_
    print('Silhouette score for k(clusters): '+str(i)+' is '+str(metrics.silhouette_score(X,kMeans_labels,metric='euclidean',sample_size=1000,random_state=200)))

start_kmeans = time.time()
km = KMeans(n_clusters=6,init='k-means++',random_state=200)
y_predicted = km.fit_predict(X)
end_kmeans = time.time()

principal_cencus_Df['cluster_kmeans']=y_predicted
print('principal_cencus_Df.head()', principal_cencus_Df.head())
print('principal_cencus_Df.shape', principal_cencus_Df.shape)
print('km.cluster_centers_', km.cluster_centers_)
print('km.cluster_centers_.shape', km.cluster_centers_.shape)
print('km.labels_', km.labels_)
print('km.inertia_', km.inertia_)

df1 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==0]
df2 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==1]
df3 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==2]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green', label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red', label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='black', label='cluster 3')
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('KMEANS Clustering')
plt.legend()
plt.show()

print('principal_cencus_Df.shape', principal_cencus_Df.shape)
principal_cencus_Df.drop('cluster_kmeans',axis=1,inplace=True)
########################################################################## DBSCAN ##########################################################################

# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
# https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html
# https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd

calculateNearestNeighbors(X)
computeEps(X)
computeMinSamples(X)

start_dbscan = time.time()
dbscan = DBSCAN(eps=2.1, min_samples=18)
dbscan.fit(X)
end_dbscan = time.time()

# The labels_ property contains the list of clusters and their respective points.
clusters = dbscan.labels_
#principal_cencus_Df['cluster_dbscan']=clusters
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
clusters = dbscan.labels_

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Visualising the clusters
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
plt.scatter(principal_cencus_Df['principal component 1'], principal_cencus_Df['principal component 2'], c=vectorizer(clusters)) # all the dark blue points were categorized as noise.
plt.title('DBSCAN Clustering')
plt.show()

########################################################################## Agglomerative Hierarchical Clustering #######################################################

dendrogrm = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

start_hierarchy = time.time()
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
end_hierarchy = time.time()
principal_cencus_Df = principal_cencus_Df.values 

# Visualising the clusters
plt.scatter(principal_cencus_Df[y_hc == 0, 0], principal_cencus_Df[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(principal_cencus_Df[y_hc == 1, 0], principal_cencus_Df[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(principal_cencus_Df[y_hc == 2, 0], principal_cencus_Df[y_hc == 2, 1], s = 50, c = 'green', label = 'Target')
plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()

########################################################################## Gaussian Mixture Model ############################################################

start_gmm = time.time()
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
end_gmm = time.time()
print('Gaussian Mixture Model labels',labels)
print('gmm.means_\n', gmm.means_)

# Visualising the clusters
plt.scatter(principal_cencus_Df[labels == 0, 0], principal_cencus_Df[labels == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(principal_cencus_Df[labels == 1, 0], principal_cencus_Df[labels == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(principal_cencus_Df[labels == 2, 0], principal_cencus_Df[labels == 2, 1], s = 50, c = 'green', label = 'Target')
plt.title('Gaussian Mixture Model')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()

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